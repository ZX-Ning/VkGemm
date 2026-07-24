#include <algorithm>
#include <print>
#include <random>

#include "core/Buffer.hpp"
#include "core/RenderPipeline.hpp"
#include "core/VulkanContext.hpp"
#include "utils.hpp"
//
#include <Eigen/Dense>

#ifdef CUBLAS
#include "cuda_ref/cublas.hpp"
#endif

constexpr uint32_t MAX_GROUP_ROWS_PER_SUBMISSION = 64;

std::vector<vk::DescriptorSetLayoutBinding> createComputeBindingLayouts(int n) {
    std::vector<vk::DescriptorSetLayoutBinding> result;
    for (int i = 0; i < n; i++) {
        result.push_back(vk::DescriptorSetLayoutBinding{
            .binding = static_cast<unsigned>(i),
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        });
    }
    return result;
}

void WriteDescriptorSet(
    std::vector<const StaticBuffer*> bufs,
    const vk::DescriptorSet& set,
    const vk::raii::Device& device
) {
    std::vector<vk::DescriptorBufferInfo> infos;
    std::vector<vk::WriteDescriptorSet> writes;
    for (size_t i = 0; i < bufs.size(); i++) {
        infos.push_back({
            .buffer = bufs[i]->getVkBuffer(),
            .offset = 0UL,
            .range = bufs[i]->size(),
        });
    }
    for (size_t i = 0; i < bufs.size(); i++) {
        writes.push_back(
            vk::WriteDescriptorSet{
                .dstSet = set,
                .dstBinding = static_cast<uint32_t>(i),
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
            }
                .setBufferInfo({infos[i]})
        );
    }
    device.updateDescriptorSets(writes, {});
}

bool varify(const Eigen::MatrixXf& a, const Eigen::MatrixXf& b, const Eigen::MatrixXf& result) {
    constexpr int ROUNDS = 200;
    static std::random_device rd;
    static std::mt19937 gen(rd());
    constexpr float EPSILON = 0.1f;
    int matSize = result.cols();

    if (matSize <= 1024) {
        Eigen::MatrixXf diff = a * b - result;
        float maxDiff = diff.cwiseAbs().maxCoeff();
        std::println("Max diff: {}", maxDiff);
        return maxDiff < EPSILON;
    }
    else {
        std::uniform_int_distribution<size_t> dist(0, matSize - 1);
        for (int i = 0; i < ROUNDS; i++) {
            size_t x = dist(gen);
            size_t y = dist(gen);
            auto left = a.row(x);
            auto right = b.col(y);
            float expected = left * right;
            float actual = result(x, y);
            if (std::abs(expected - actual) > EPSILON) {
                return false;
            }
        }
        return true;
    }
}

void runCpu(
    Eigen::MatrixX<Eigen::half>& mat1,
    Eigen::MatrixX<Eigen::half>& mat2,
    Eigen::MatrixXf& mat3
) {
    std::println("Begin CPU compute");
    size_t time1 = getTimestampMs();
    mat3 += mat1.cast<float>() * mat2.cast<float>();
    std::println("CPU compute Done. Time: {} ms", getTimestampMs() - time1);
}

bool testResult(const Eigen::MatrixXf& result, const Eigen::MatrixXf& expected) {
    constexpr float EPSILON = 0.1f;
    Eigen::MatrixXf diff = expected - result;
    float maxDiff = diff.cwiseAbs().maxCoeff();
    // std::println("Max diff: {}", maxDiff);
    return maxDiff < EPSILON;
}

auto genData(int matSize) {
    std::println("Preparing data...");
    Eigen::MatrixX<Eigen::half> mat1(matSize, matSize);
    Eigen::MatrixX<Eigen::half> mat2(matSize, matSize);
    Eigen::MatrixXf mat3(matSize, matSize);
    mat1.setRandom();
    mat2.setRandom();
    mat3.setZero();
    return std::tuple{std::move(mat1), std::move(mat2), std::move(mat3)};
}

int run(
    VulkanContext& ctx,
    const std::string& spvPath,
    uint32_t tileSize,
    uint32_t matSize,
    Eigen::MatrixX<Eigen::half>& mat1,
    Eigen::MatrixX<Eigen::half>& mat2,
    Eigen::MatrixXf& mat3,
    std::array<uint32_t, 3> numthreads

) {
    std::println("Running kernel: {}, Matrix size: {}x{}", spvPath, matSize, matSize);
    int bufSize = matSize * matSize;
    auto bindslayouts = createComputeBindingLayouts(3);
    auto setLayout = ctx.device.createDescriptorSetLayout(
        vk::DescriptorSetLayoutCreateInfo{}
            .setBindings(bindslayouts)
    );

    vk::PushConstantRange pcRange{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = 3 * sizeof(uint32_t),
    };

    auto layout = ctx.device.createPipelineLayout(
        vk::PipelineLayoutCreateInfo{}
            .setSetLayouts({*setLayout})
            .setPushConstantRanges({pcRange})
    );
    auto pipeline = createComputePipeline(
        ctx,
        {layout, readFile(spvPath)},
        numthreads
    );

    auto matABuf = BufferFactory::createStaticBuffer(
        BufferFactory::Type::Storage,
        *ctx.allocator,
        bufSize * sizeof(Eigen::half)
    );
    auto matBBuf = BufferFactory::createStaticBuffer(
        BufferFactory::Type::Storage,
        *ctx.allocator,
        bufSize * sizeof(Eigen::half)
    );
    auto matCBuf = BufferFactory::createStaticBuffer(
        BufferFactory::Type::Storage,
        *ctx.allocator,
        bufSize * sizeof(float)
    );

    auto sets = ctx.device.allocateDescriptorSets({
        .descriptorPool = ctx.descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &*setLayout,
    });

    WriteDescriptorSet(
        {
            matABuf.get(),
            matBBuf.get(),
            matCBuf.get(),
        },
        *sets[0],
        ctx.device
    );

    std::println("Loading data to VRAM...");
    ctx.loadingCmdBuffer.begin({});
    {
        matABuf->load(
            std::span((uint8_t*)mat1.data(), bufSize * sizeof(Eigen::half)),
            ctx.loadingCmdBuffer
        );
        matBBuf->load(
            std::span((uint8_t*)mat2.data(), bufSize * sizeof(Eigen::half)),
            ctx.loadingCmdBuffer
        );
        matCBuf->load(
            std::span((uint8_t*)mat3.data(), bufSize * sizeof(float)),
            ctx.loadingCmdBuffer
        );
        ctx.loadingCmdBuffer.end();
    }
    ctx.queue.submit({vk::SubmitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = &*ctx.loadingCmdBuffer,
    }});
    ctx.device.waitIdle();
    matABuf->deleteStaging();
    matBBuf->deleteStaging();
    matCBuf->deleteStaging();

    auto cmdBufs =
        ctx.device.allocateCommandBuffers({
            .commandPool = ctx.commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1,
        });
    assert(cmdBufs.size() == 1);
    auto& cmd = cmdBufs[0];

    std::println("Begin Compute");
    // assert(MAT_SIZE % TILE_SIZE == 0);
    const uint32_t groupCount = (matSize + tileSize - 1) / tileSize;
    const auto time1 = getTimestampMs();

    for (uint32_t baseGroupY = 0; baseGroupY < groupCount;
         baseGroupY += MAX_GROUP_ROWS_PER_SUBMISSION) {
        const uint32_t groupRows = std::min(
            MAX_GROUP_ROWS_PER_SUBMISSION,
            groupCount - baseGroupY
        );

        cmd.begin({});
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->pipeline);
        cmd.pushConstants(
            layout,
            vk::ShaderStageFlagBits::eCompute,
            0,
            vk::ArrayProxy<const uint32_t>{matSize, matSize, matSize}
        );
        cmd.bindDescriptorSets(
            vk::PipelineBindPoint::eCompute,
            layout,
            0,
            std::array{*sets[0]},
            nullptr
        );
        cmd.dispatchBase(
            0,
            baseGroupY,
            0,
            groupCount,
            groupRows,
            1
        );
        // cmd.dispatch(groupCount, groupCount, 1);
        cmd.end();

        const vk::SubmitInfo submitInfo{
            .waitSemaphoreCount = 0,
            .commandBufferCount = 1,
            .pCommandBuffers = &*cmd,
            .signalSemaphoreCount = 0
        };
        ctx.queue.submit(submitInfo, nullptr);
        ctx.queue.waitIdle();
    }
    std::println("Compute Done. Time used: {} ms", getTimestampMs() - time1);
    mat3.setZero();
    matCBuf->readBackSyncDangerous(ctx, (uint8_t*)mat3.data());
    std::println("Result read back. Begin validation.");
    return 0;
}

int main(int argc, char** argv) {
    size_t matSize = 1 << 14;
    if (argc == 2) {
        matSize = std::stoi(argv[1]);
    }
    VulkanContext ctx{};
    auto [mat1, mat2, mat3] = genData(matSize);
    Eigen::MatrixXf mat3_ref(mat3);
    std::println("--------------------------------------");
#ifdef CUBLAS
    runCuBlas(matSize, mat1, mat2, mat3_ref);
#else
    runCpu(mat1, mat2, mat3_ref);
#endif
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    std::println("--------------------------------------");
    mat3.setZero();
    run(
        ctx,
        "shaders/gemm.spv",
        32,
        matSize,
        mat1,
        mat2,
        mat3,
        {32, 32, 1}
    );
    ctx.device.waitIdle();
    if (!testResult(mat3, mat3_ref)) {
        std::println(stderr, "not match!");
        exit(1);
    }
    std::println("Test Pass. Done.");
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    std::println("--------------------------------------");
    mat3.setZero();
    run(
        ctx,
        "shaders/gemm_coopmat.spv",
        16,
        matSize,
        mat1,
        mat2,
        mat3,
        {ctx.subgroupSize, 1, 1}
    );
    ctx.device.waitIdle();
    if (!testResult(mat3, mat3_ref)) {
        std::println(stderr, "not match!");
        exit(1);
    }
    std::println("Test Pass. Done.");
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    std::println("--------------------------------------");
    mat3.setZero();
    run(
        ctx,
        "shaders/gemm_coopmat_opt.spv",
        32,
        matSize,
        mat1,
        mat2,
        mat3,
        {ctx.subgroupSize, 1, 1}
    );
    ctx.device.waitIdle();
    if (!testResult(mat3, mat3_ref)) {
        std::println(stderr, "not match!");
        exit(1);
    }
    std::println("Test Pass. Done.");
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    std::println("--------------------------------------");
    mat3.setZero();
    run(
        ctx,
        "shaders/gemm_coopmat_tiled.spv",
        64,
        matSize,
        mat1,
        mat2,
        mat3,
        {ctx.subgroupSize, 4, 4}
    );
    ctx.device.waitIdle();
    if (!testResult(mat3, mat3_ref)) {
        std::println(stderr, "not match!");
        exit(1);
    }
    std::println("Test Pass. Done.");
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    std::println("--------------------------------------");
    mat3.setZero();
    run(
        ctx,
        "shaders/gemm_coopmat_tiled_opt.spv",
        64,
        matSize,
        mat1,
        mat2,
        mat3,
        {ctx.subgroupSize, 2, 2}
    );
    if (!testResult(mat3, mat3_ref)) {
        std::println(stderr, "not match!");
        exit(1);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    std::println("--------------------------------------");
    mat3.setZero();
    run(
        ctx,
        "shaders/gemm_coopmat_tiled_opt2.spv",
        64,
        matSize,
        mat1,
        mat2,
        mat3,
        {ctx.subgroupSize, 2, 2}
    );
    if (!testResult(mat3, mat3_ref)) {
        std::println(stderr, "not match!");
        exit(1);
    }
    std::println("Test Pass. Done.");

    return 0;
}
