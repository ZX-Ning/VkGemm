#include <fmt/format.h>

#include <algorithm>
#include <random>

#include "core/Buffer.hpp"
#include "core/RenderPipeline.hpp"
#include "core/VulkanContext.hpp"
#include "shader/SlangShaderCompiler.hpp"
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

// bool varify(const Eigen::MatrixXf& a, const Eigen::MatrixXf& b, const Eigen::MatrixXf& result) {
//     constexpr int ROUNDS = 200;
//     static std::random_device rd;
//     static std::mt19937 gen(rd());
//     constexpr float EPSILON = 0.1f;
//     int matSize = result.cols();

//     if (matSize <= 1024) {
//         Eigen::MatrixXf diff = a * b - result;
//         float maxDiff = diff.cwiseAbs().maxCoeff();
//         fmt::println("Max diff: {}", maxDiff);
//         return maxDiff < EPSILON;
//     }
//     else {
//         std::uniform_int_distribution<size_t> dist(0, matSize - 1);
//         for (int i = 0; i < ROUNDS; i++) {
//             size_t x = dist(gen);
//             size_t y = dist(gen);
//             auto left = a.row(x);
//             auto right = b.col(y);
//             float expected = left * right;
//             float actual = result(x, y);
//             if (std::abs(expected - actual) > EPSILON) {
//                 return false;
//             }
//         }
//         return true;
//     }
// }

void runCpu(
    Eigen::MatrixX<Eigen::half>& mat1,
    Eigen::MatrixX<Eigen::half>& mat2,
    Eigen::MatrixXf& mat3
) {
    fmt::println("Begin CPU compute");
    size_t time1 = getTimestampMs();
    mat3 += mat1.cast<float>() * mat2.cast<float>();
    fmt::println("CPU compute Done. Time: {} ms", getTimestampMs() - time1);
}

bool testResult(const Eigen::MatrixXf& result, const Eigen::MatrixXf& expected) {
    constexpr float EPSILON = 0.1f;
    Eigen::MatrixXf diff = expected - result;
    float maxDiff = diff.cwiseAbs().maxCoeff();
    // fmt::println("Max diff: {}", maxDiff);
    return maxDiff < EPSILON;
}

auto genData(int matSize) {
    fmt::println("Preparing data...");
    Eigen::MatrixX<Eigen::half> mat1(matSize, matSize);
    Eigen::MatrixX<Eigen::half> mat2(matSize, matSize);
    Eigen::MatrixXf mat3(matSize, matSize);
    mat1.setRandom();
    mat2.setRandom();
    mat3.setRandom();
    return std::tuple{std::move(mat1), std::move(mat2), std::move(mat3)};
}

size_t run(
    VulkanContext& ctx,
    std::span<const uint32_t> spv,
    uint32_t tileSize,
    uint32_t matSize,
    Eigen::MatrixX<Eigen::half>& mat1,
    Eigen::MatrixX<Eigen::half>& mat2,
    Eigen::MatrixXf& mat3,
    std::array<uint32_t, 3> numthreads

) {
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
        {layout, spv},
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

    fmt::println("Loading data to VRAM...");
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

    fmt::println("Begin Compute");
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
    size_t timeUsed = getTimestampMs() - time1;
    fmt::println("Compute Done. Time used: {} ms. Reading back result...", timeUsed);
    matCBuf->readBackSyncDangerous(ctx, (uint8_t*)mat3.data());
    fmt::println("Result read back. Begin validation.");
    ctx.device.waitIdle();
    return timeUsed;
}

struct TestOption {
    const char* shaderPath;
    uint32_t tileSize;
    std::array<uint32_t, 3> numthreads;
    uint32_t matSizeMultiple;
    bool requiresCooperativeMatrix2 = false;
};

constexpr TestOption TESTS[] = {
    {"shaders/gemm/experiment.slang", 64, {32, 4, 2}, 64},
    {"shaders/gemm/coopmat_cm2_64x64.slang",
     64,
     {256, 1, 1},
     64,
     true},
    {"shaders/gemm/coopmat_64x64_4acc.slang",
     64,
     {32, 2, 2},
     64},
    {"shaders/gemm/coopmat_128x128_16acc.slang",
     128,
     {32, 2, 2},
     128},
    {"shaders/gemm/plain.slang", 32, {32, 32, 1}, 1},
    {"shaders/gemm/coopmat_plain.slang", 16, {32, 1, 1}, 16},
    {"shaders/gemm/coopmat_plain_4acc.slang",
     32,
     {32, 1, 1},
     32},
    {"shaders/gemm/coopmat_tiled_1acc.slang",
     64,
     {32, 4, 4},
     1},
    {"shaders/gemm/coopmat_4acc_simpleload.slang",
     64,
     {32, 2, 2},
     1},
};

int main(int argc, char** argv) {
    constexpr int COOLDOWN_TIME = 5000;
    size_t matSize = 1 << 14;
    if (argc == 2) {
        matSize = std::stoi(argv[1]);
    }
    VulkanContext ctx{};
    SlangShaderCompiler shaderCompiler;
    auto [mat1, mat2, mat3] = genData(matSize);
    Eigen::MatrixXf originMat3(mat3);
    Eigen::MatrixXf resultRef(mat3);
    std::vector<std::tuple<std::string, int>> results;
    fmt::println("{:-^60}", "");
#ifdef CUBLAS
    size_t refTime = runCuBlas(matSize, mat1, mat2, resultRef);
    results.push_back({"cuBLAS (reference)", refTime});
#else
    runCpu(mat1, mat2, mat3_ref);
#endif
    if (ctx.subgroupSize != 32) {
        fmt::println(stderr, "Subgroup size is {} not 32 ! Abort.", ctx.subgroupSize);
        exit(1);
    }

    for (const auto& test : TESTS) {
        if (matSize % test.matSizeMultiple != 0) {
            fmt::println(
                "Skipping kernel: {} (matrix size {} is not a multiple of {})",
                test.shaderPath,
                matSize,
                test.matSizeMultiple
            );
            continue;
        }

        if (test.requiresCooperativeMatrix2 &&
            !ctx.supportsCooperativeMatrix2()) {
            fmt::println(
                "Skipping kernel: {} (cooperative matrix 2 unsupported)",
                test.shaderPath
            );
            continue;
        }

        fmt::println("Cooldown {} ms...", COOLDOWN_TIME);
        std::this_thread::sleep_for(std::chrono::milliseconds(COOLDOWN_TIME));
        fmt::println("{:-^60}", "");
        fmt::println("Compiling kernel: {}", test.shaderPath);
        const auto shaderSource = readFile(test.shaderPath);
        const auto spv = shaderCompiler.genSpirv(std::string_view{
            reinterpret_cast<const char*>(shaderSource.data()),
            shaderSource.size(),
        });
        fmt::println(
            "Running kernel: {}, Matrix size: {}x{}",
            test.shaderPath,
            matSize,
            matSize
        );
        mat3 = originMat3;
        uint64_t time = run(
            ctx,
            spv,
            test.tileSize,
            matSize,
            mat1,
            mat2,
            mat3,
            test.numthreads
        );
        if (!testResult(mat3, resultRef)) {
            fmt::println(stderr, "not match!!!");
            exit(1);
        }
        results.push_back({test.shaderPath, time});
        fmt::println("Test Pass.");
    }

    fmt::println("");
    fmt::println("{:-^60}", "RESULT");
    std::ranges::sort(
        results,
        [](const auto& r1, const auto& r2) { return std::get<1>(r1) < std::get<1>(r2); }
    );
    for (auto& [spvPath, time] : results) {
#ifdef CUBLAS
        fmt::println("{:<50} {:>15} ms ({:.2f}%)", spvPath + ":", time, 100.0 * refTime / time);
#else
        fmt::println("{:<50}: {:>15} ms", spvPath, time);
#endif
    }
    return 0;
}
