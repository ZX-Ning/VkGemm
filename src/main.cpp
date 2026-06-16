#include <print>
#include <ranges>

#include "core/Buffer.hpp"
#include "core/RenderPipeline.hpp"
#include "core/VulkanContext.hpp"
#include "utils.hpp"

constexpr int VEC_SIZE = 4096;
constexpr int GROUP_SIZE = 256;

int main() {
    VulkanContext ctx{};
    std::array bindingsLayout = {
        vk::DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 2,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute
        }
    };
    vk::raii::DescriptorSetLayout bindLayout(
        ctx.device,
        {
            .bindingCount = 3,
            .pBindings = bindingsLayout.data(),
        }
    );
    vk::PushConstantRange pcRange{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(uint32_t),
    };

    vk::raii::PipelineLayout layout(
        ctx.device,
        {
            .setLayoutCount = 1,
            .pSetLayouts = &*bindLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pcRange,
        }
    );
    auto pipeline = createComputePipeline(
        ctx,
        {.layout = layout, .shaderSpv = readFile("shaders/compute.spv")}
    );

    auto vec1Buffer =
        BufferFactory::createStaticBuffer(BufferFactory::Type::Storage, *ctx.allocator, VEC_SIZE * sizeof(float));
    auto vec2Buffer =
        BufferFactory::createStaticBuffer(BufferFactory::Type::Storage, *ctx.allocator, VEC_SIZE * sizeof(float));
    auto vecOutBuffer =
        BufferFactory::createStaticBuffer(BufferFactory::Type::Storage, *ctx.allocator, VEC_SIZE * sizeof(float));

    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = ctx.descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &*bindLayout
    };
    auto sets =
        ctx.device.allocateDescriptorSets(allocInfo);

    vk::DescriptorBufferInfo bufInfo1 = {
        .buffer = vec1Buffer->getVkBuffer(),
        .offset = 0,
        .range = VEC_SIZE * sizeof(float)
    };
    vk::DescriptorBufferInfo bufInfo2 = {
        .buffer = vec2Buffer->getVkBuffer(),
        .offset = 0,
        .range = VEC_SIZE * sizeof(float)
    };
    vk::DescriptorBufferInfo bufInfoOut = {
        .buffer = vecOutBuffer->getVkBuffer(),
        .offset = 0,
        .range = VEC_SIZE * sizeof(float)
    };
    ctx.device.updateDescriptorSets(
        std::array{
            vk::WriteDescriptorSet{
                .dstSet = *sets[0],
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &bufInfo1
            },
            vk::WriteDescriptorSet{
                .dstSet = *sets[0],
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &bufInfo2
            },
            vk::WriteDescriptorSet{
                .dstSet = *sets[0],
                .dstBinding = 2,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &bufInfoOut
            }
        },
        {}
    );

    std::vector<float> vec1 =
        std::ranges::views::iota(0, VEC_SIZE) |
        std::ranges::views::transform([](int x) { return static_cast<float>(x); }) |
        std::ranges::to<std::vector>();

    std::vector<float> vec2(vec1);
    assert(vec1.size() == VEC_SIZE);
    assert(vec2.size() == VEC_SIZE);

    ctx.loadingCmdBuffer.begin({});
    vec1Buffer->load(asRawBytes(vec1), ctx.loadingCmdBuffer);
    vec2Buffer->load(asRawBytes(vec2), ctx.loadingCmdBuffer);
    ctx.loadingCmdBuffer.end();
    ctx.queue.submit({vk::SubmitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = &*ctx.loadingCmdBuffer,
    }});
    ctx.device.waitIdle();
    vec1Buffer->deleteStaging();
    vec2Buffer->deleteStaging();

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
    cmd.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, vk::ArrayProxy<const uint32_t>{VEC_SIZE});
    cmd.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute,
        layout,
        0,
        std::array{*sets[0]},
        nullptr
    );
    int groupCount = VEC_SIZE / GROUP_SIZE;
    cmd.dispatch(groupCount, 1, 1);
    vk::BufferMemoryBarrier2 barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        .buffer = vecOutBuffer->getVkBuffer(),
        .offset = 0,
        .size = VEC_SIZE * sizeof(float)
    };
    cmd.pipelineBarrier2(
        vk::DependencyInfo{
            .dependencyFlags = {},
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &barrier,
        }
    );
    auto result = vecOutBuffer->readBack(cmd);
    cmd.end();

    const vk::SubmitInfo submitInfo{
        .waitSemaphoreCount = 0,
        .commandBufferCount = 1,
        .pCommandBuffers = &*cmd,
        .signalSemaphoreCount = 0
    };
    ctx.queue.submit(submitInfo, nullptr);
    ctx.device.waitIdle();
    std::println("Compute Done");

    std::span<float> resultCast{(float*)result.data(), VEC_SIZE};
    for (int i = 0; i < VEC_SIZE; i++) {
        if (std::abs(resultCast[i] - 2 * i) > 0.0001) {
            throw std::runtime_error("Calculation Wrong!");
        }
    }
    return 0;
}
