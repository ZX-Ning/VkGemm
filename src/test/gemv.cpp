#include <fmt/format.h>

#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <string_view>

#include "../core/Buffer.hpp"
#include "../core/RenderPipeline.hpp"
#include "../core/VulkanContext.hpp"
#include "../shader/SlangShaderCompiler.hpp"
#include "../utils.hpp"
#include "./common.hpp"

namespace {

constexpr char SHADER_PATH[] = "shaders/gemv/gemv.slang";
constexpr uint32_t DEFAULT_M = 1U << 13;
constexpr uint32_t DEFAULT_N = 1U << 15;
constexpr uint32_t NUM_THREADS = 1024;

size_t runCpu(
    const Eigen::MatrixXf& matA,
    const Eigen::VectorXf& vecX,
    Eigen::VectorXf& vecY
) {
    const size_t start = getTimestampMs();
    vecY += matA * vecX;
    return getTimestampMs() - start;
}

bool testResult(
    const Eigen::VectorXf& result,
    const Eigen::VectorXf& expected
) {
    constexpr float EPSILON = 0.01f;
    const float maxDiff = (expected - result).cwiseAbs().maxCoeff();
    fmt::println("Max diff: {}", maxDiff);
    return maxDiff < EPSILON;
}

size_t runGpu(
    VulkanContext& ctx,
    std::span<const uint32_t> spv,
    const Eigen::MatrixXf& matA,
    const Eigen::VectorXf& vecX,
    Eigen::VectorXf& vecY
) {
    const auto bindings = createComputeBindingLayouts(3);
    auto setLayout = ctx.device.createDescriptorSetLayout(
        vk::DescriptorSetLayoutCreateInfo{}.setBindings(bindings)
    );

    const vk::PushConstantRange pcRange{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = 2 * sizeof(uint32_t),
    };
    auto layout = ctx.device.createPipelineLayout(
        vk::PipelineLayoutCreateInfo{}
            .setSetLayouts({*setLayout})
            .setPushConstantRanges({pcRange})
    );

    const std::array numThreads{NUM_THREADS, 1U, 1U};
    auto pipeline = createComputePipeline(ctx, {layout, spv}, numThreads);

    auto matABuf = BufferFactory::createStaticBuffer(
        BufferFactory::Type::Storage,
        *ctx.allocator,
        static_cast<size_t>(matA.size()) * sizeof(float)
    );
    auto vecXBuf = BufferFactory::createStaticBuffer(
        BufferFactory::Type::Storage,
        *ctx.allocator,
        static_cast<size_t>(vecX.size()) * sizeof(float)
    );
    auto vecYBuf = BufferFactory::createStaticBuffer(
        BufferFactory::Type::Storage,
        *ctx.allocator,
        static_cast<size_t>(vecY.size()) * sizeof(float)
    );

    auto sets = ctx.device.allocateDescriptorSets({
        .descriptorPool = ctx.descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &*setLayout,
    });
    WriteDescriptorSet(
        {matABuf.get(), vecXBuf.get(), vecYBuf.get()},
        *sets[0],
        ctx.device
    );

    ctx.loadingCmdBuffer.begin({});
    matABuf->load(
        std::span(
            reinterpret_cast<const uint8_t*>(matA.data()),
            static_cast<size_t>(matA.size()) * sizeof(float)
        ),
        ctx.loadingCmdBuffer
    );
    vecXBuf->load(
        std::span(
            reinterpret_cast<const uint8_t*>(vecX.data()),
            static_cast<size_t>(vecX.size()) * sizeof(float)
        ),
        ctx.loadingCmdBuffer
    );
    vecYBuf->load(
        std::span(
            reinterpret_cast<const uint8_t*>(vecY.data()),
            static_cast<size_t>(vecY.size()) * sizeof(float)
        ),
        ctx.loadingCmdBuffer
    );
    ctx.loadingCmdBuffer.end();
    ctx.queue.submit({vk::SubmitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = &*ctx.loadingCmdBuffer,
    }});
    ctx.device.waitIdle();
    matABuf->deleteStaging();
    vecXBuf->deleteStaging();
    vecYBuf->deleteStaging();

    auto cmdBufs = ctx.device.allocateCommandBuffers({
        .commandPool = ctx.commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    });
    auto& cmd = cmdBufs[0];

    const std::array shape{
        static_cast<uint32_t>(matA.rows()),
        static_cast<uint32_t>(matA.cols()),
    };
    cmd.begin({});
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->pipeline);
    cmd.pushConstants(
        layout,
        vk::ShaderStageFlagBits::eCompute,
        0,
        vk::ArrayProxy<const uint32_t>{shape}
    );
    cmd.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute,
        layout,
        0,
        std::array{*sets[0]},
        nullptr
    );

    const size_t start = getTimestampMs();
    const uint32_t rowsPerGroup = NUM_THREADS / ctx.subgroupSize;
    const uint32_t groupCount = (shape[0] + rowsPerGroup - 1) / rowsPerGroup;
    cmd.dispatch(groupCount, 1, 1);
    cmd.end();
    ctx.queue.submit({vk::SubmitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = &*cmd,
    }});
    ctx.queue.waitIdle();
    const size_t timeUsed = getTimestampMs() - start;

    vecYBuf->readBackSyncDangerous(
        ctx,
        reinterpret_cast<uint8_t*>(vecY.data())
    );
    return timeUsed;
}

}  // namespace

int main(int argc, char** argv) {
    fmt::println("RUNNING GEMV TEST");

    uint32_t m = DEFAULT_M;
    uint32_t n = DEFAULT_N;
    if (argc == 3) {
        try {
            m = static_cast<uint32_t>(std::stoul(argv[1]));
            n = static_cast<uint32_t>(std::stoul(argv[2]));
        }
        catch (...) {
        }
    }

    fmt::println("Preparing data: A={}x{}, x={}, y={}", m, n, n, m);
    Eigen::MatrixXf matA(m, n);
    Eigen::VectorXf vecX(n);
    Eigen::VectorXf vecY(m);
    matA.setRandom();
    vecX.setRandom();
    vecY.setRandom();

    Eigen::VectorXf expected = vecY;
    const size_t cpuTime = runCpu(matA, vecX, expected);

    VulkanContext ctx{};
    SlangShaderCompiler shaderCompiler;
    const auto shaderSource = readFile(SHADER_PATH);
    const auto spv = shaderCompiler.genSpirv(std::string_view{
        reinterpret_cast<const char*>(shaderSource.data()),
        shaderSource.size(),
    });
    const size_t gpuTime = runGpu(ctx, spv, matA, vecX, vecY);

    if (!testResult(vecY, expected)) {
        fmt::println(stderr, "Test failed: GPU result does not match CPU result.");
        return 1;
    }

    fmt::println("Test passed. CPU: {} ms, GPU: {} ms", cpuTime, gpuTime);
    return 0;
}
