#include <fmt/format.h>
#include <fmt/os.h>

#include <algorithm>
#include <filesystem>
#include <random>
#include <thread>

#include "../core/Buffer.hpp"
#include "../core/RenderPipeline.hpp"
#include "../core/VulkanContext.hpp"
#include "../shader/SlangShaderCompiler.hpp"
#include "../utils.hpp"
//
#include <Eigen/Dense>
#include <argparse/argparse.hpp>

#include "./common.hpp"
#ifdef CUDA_REF
#include "../cuda_ref/cuda_ref.hpp"
#endif

constexpr uint32_t MAX_GROUP_ROWS_PER_SUBMISSION = 256;

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

    // for (uint32_t baseGroupY = 0; baseGroupY < groupCount;
    //      baseGroupY += MAX_GROUP_ROWS_PER_SUBMISSION) {
    // const uint32_t groupRows = std::min(
    //     MAX_GROUP_ROWS_PER_SUBMISSION,
    //     groupCount - baseGroupY
    // );

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
    // cmd.dispatchBase(
    //     0,
    //     baseGroupY,
    //     0,
    //     groupCount,
    //     groupRows,
    //     1
    // );
    cmd.dispatch(groupCount, groupCount, 1);
    cmd.end();

    const vk::SubmitInfo submitInfo{
        .waitSemaphoreCount = 0,
        .commandBufferCount = 1,
        .pCommandBuffers = &*cmd,
        .signalSemaphoreCount = 0
    };
    ctx.queue.submit(submitInfo, nullptr);
    ctx.queue.waitIdle();
    // }
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
    {
        "shaders/gemm/coopmat_m64n64k64_4acc.slang",
        64,
        {32, 2, 2},
        64,
    },
    {
        "shaders/gemm/coopmat_m64n64k64_2acc.slang",
        64,
        {32, 4, 2},
        64,
    },
    {
        "shaders/gemm/coopmat_cm2_64x64.slang",
        64,
        {256, 1, 1},
        64,
        true,
    },
    {
        "shaders/gemm/coopmat_m64n64k32_4acc.slang",
        64,
        {32, 2, 2},
        64,
    },
    {
        "shaders/gemm/coopmat_m128n128k32_16acc.slang",
        128,
        {32, 2, 2},
        128,
    },
    // {"shaders/gemm/plain.slang", 32, {32, 32, 1}, 1},
    // {"shaders/gemm/coopmat_plain.slang", 16, {32, 1, 1}, 16},
};

int main(int argc, char** argv) {
    argparse::ArgumentParser program("test_gemm");

    program.add_argument("size")
        .help("Size of the matrix (M=N=K=size)")
        .default_value(1U << 14)
        .scan<'u', uint32_t>();

    program.add_argument("--kernel")
        .help("Which kernel to run");

    program.add_argument("--cooldown")
        .default_value(5000)
        .scan<'d', int>()
        .help("Cooldown time between test");

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }
    uint32_t matSize = program.get<uint32_t>("size");
    std::optional<std::string> kernelName = program.present("--kernel");

    fmt::println("RUNNING GEMM TEST");

    int cooldownTime = program.get<int>("--cooldown");

    if (argc == 2) {
        matSize = std::stoi(argv[1]);
    }
    VulkanContext ctx{};
    SlangShaderCompiler shaderCompiler;
    auto [mat1, mat2, mat3] = genData(matSize);
    Eigen::MatrixXf originMat3(mat3);
    Eigen::MatrixXf resultRef(mat3);
    std::vector<std::tuple<std::string, int>> results;

#ifdef CUDA_REF
    size_t refTime = runCuBlas(matSize, mat1, mat2, resultRef);  // warmup
    results.emplace_back("cuBLAS (ref)", refTime);

    if (kernelName.has_value() && *kernelName == "cuda") {
        fmt::println("Cooldown {} ms...", cooldownTime);
        std::this_thread::sleep_for(std::chrono::milliseconds(cooldownTime));
        fmt::println("{:-^60}", "");
        mat3 = originMat3;
        size_t cudaExp2Time =
            runCudaExperiment2(matSize, mat1, mat2, mat3);
        if (!testResult(mat3, resultRef)) {
            fmt::println(stderr, "CUDA Experiment2 result does not match cuBLAS!");
            exit(1);
        }
        results.push_back({"experiment2_wmma.cu", cudaExp2Time});

        fmt::println("Cooldown {} ms...", cooldownTime);
        std::this_thread::sleep_for(std::chrono::milliseconds(cooldownTime));
        fmt::println("{:-^60}", "");
        mat3 = originMat3;
        size_t cuda128x128Time =
            runCuda128x12816Acc(matSize, mat1, mat2, mat3);
        if (!testResult(mat3, resultRef)) {
            fmt::println(
                stderr,
                "CUDA 128x128 16ACC result does not match cuBLAS!"
            );
            exit(1);
        }
        results.push_back({"128x128_16acc_wmma.cu", cuda128x128Time});
    }
#else
    runCpu(mat1, mat2, resultRef);
#endif
    if (ctx.subgroupSize != 32) {
        fmt::println(stderr, "Subgroup size is {} not 32 ! Abort.", ctx.subgroupSize);
        exit(1);
    }

    for (const auto& test : TESTS) {
        if (kernelName.has_value() && *kernelName != test.shaderPath) {
            continue;
        }
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

        fmt::println("Cooldown {} ms...", cooldownTime);
        std::this_thread::sleep_for(std::chrono::milliseconds(cooldownTime));
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

    fmt::println("Writing log...");

    std::filesystem::create_directory("logs");
    auto out = fmt::output_file(
        fmt::format("logs/log-{}.txt", getTimestampMs()),
        fmt::file::CREATE | fmt::file::WRONLY
    );
    out.print("Device: {}\n", ctx.getDeviceName());
    out.print("Matrix: {}x{}\n", matSize, matSize);
    std::ranges::sort(
        results,
        [](const auto& r1, const auto& r2) { return std::get<1>(r1) < std::get<1>(r2); }
    );
    for (auto& [spvPath, time] : results) {
#ifdef CUDA_REF
        out.print("{:<50} {:>15} ms ({:.2f}%)\n", spvPath + ":", time, 100.0 * refTime / time);
#else
        out.print("{:<50}: {:>15} ms\n", spvPath, time);
#endif
    }
    fmt::println("Done");
    return 0;
}
