#include <print>
#include <ranges>

#include "core/Buffer.hpp"
#include "core/RenderPipeline.hpp"
#include "core/VulkanContext.hpp"
#include "utils.hpp"
//
#include <Eigen/Dense>

constexpr size_t MAT_SIZE = 15000;
constexpr size_t BUF_SIZE_FLOAT = MAT_SIZE * MAT_SIZE;
constexpr int TILE_SIZE = 16;

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

int main() {
    VulkanContext ctx{};
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
        {.layout = layout, .shaderSpv = readFile("shaders/gemm.spv")}
    );

    auto matABuf =
        BufferFactory::createStaticBuffer(BufferFactory::Type::Storage, *ctx.allocator, BUF_SIZE_FLOAT * sizeof(float));
    auto matBBuf =
        BufferFactory::createStaticBuffer(BufferFactory::Type::Storage, *ctx.allocator, BUF_SIZE_FLOAT * sizeof(float));
    auto matCBuf =
        BufferFactory::createStaticBuffer(BufferFactory::Type::Storage, *ctx.allocator, BUF_SIZE_FLOAT * sizeof(float));

    auto sets = ctx.device.allocateDescriptorSets({
        .descriptorPool = ctx.descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &*setLayout,
    });

    WriteDescriptorSet(
        {matABuf.get(), matBBuf.get(), matCBuf.get()}, *sets[0], ctx.device
    );

    Eigen::MatrixXf mat1(MAT_SIZE, MAT_SIZE);
    Eigen::MatrixXf mat2(MAT_SIZE, MAT_SIZE);
    mat1.setRandom();
    mat2.setRandom();

    auto time1 = getTimestampMs();
    std::println("Loading data to VRAM...");
    ctx.loadingCmdBuffer.begin({});
    matABuf->load(std::span((uint8_t*)mat1.data(), BUF_SIZE_FLOAT * sizeof(float)), ctx.loadingCmdBuffer);
    matBBuf->load(std::span((uint8_t*)mat2.data(), BUF_SIZE_FLOAT * sizeof(float)), ctx.loadingCmdBuffer);
    ctx.loadingCmdBuffer.end();
    ctx.queue.submit({vk::SubmitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = &*ctx.loadingCmdBuffer,
    }});
    ctx.device.waitIdle();
    matABuf->deleteStaging();
    matBBuf->deleteStaging();
    
    auto cmdBufs = ctx.device.allocateCommandBuffers({
        .commandPool = ctx.commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    });
    assert(cmdBufs.size() == 1);
    auto& cmd = cmdBufs[0];
    
    std::println("Begin Compute");
    cmd.begin({});

    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->pipeline);
    cmd.pushConstants(
        layout,
        vk::ShaderStageFlagBits::eCompute,
        0,
        vk::ArrayProxy<const uint32_t>{MAT_SIZE, MAT_SIZE, MAT_SIZE}
    );
    cmd.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute,
        layout,
        0,
        std::array{*sets[0]},
        nullptr
    );
    int groupCount = std::ceil((float)MAT_SIZE / TILE_SIZE);
    cmd.dispatch(groupCount, groupCount, 1);
    cmd.end();

    const vk::SubmitInfo submitInfo{
        .waitSemaphoreCount = 0,
        .commandBufferCount = 1,
        .pCommandBuffers = &*cmd,
        .signalSemaphoreCount = 0
    };
    ctx.queue.submit(submitInfo, nullptr);
    ctx.device.waitIdle();
    std::println("Compute Done. Reading back result...");
    std::println("Time used: {} ms", getTimestampMs() - time1);
    auto result = matCBuf->readBackSync<float>(ctx);
    std::println("Result read back. Begin CPU compute.");

    auto time2 = getTimestampMs();
    Eigen::MatrixXf expected = mat1 * mat2;
    std::println("CPU Calculation Done. Time used: {} ms. Validating...", getTimestampMs() - time2);

#pragma omp parallel for
    for (size_t i = 0; i < BUF_SIZE_FLOAT; i++) {
        if (std::abs(result[i] - expected.data()[i]) > 0.1) {
            std::println("Calculation Wrong! i = {}, expected {}, found: {}", i, expected.data()[i], result[i]);
            exit(1);
        }
    }
    std::println("Done.");

    return 0;
}
