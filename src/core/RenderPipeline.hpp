#ifndef RENDERPIPELINE_HPP
#define RENDERPIPELINE_HPP

#include <memory>
#include <span>
#include <vulkan/vulkan_raii.hpp>
// #include "VulkanContext.hpp"

struct VulkanContext;

struct Pipeline {
    vk::raii::Pipeline pipeline{nullptr};
};

struct GraphicsPipelineDesc {
    vk::PipelineLayout layout;
    std::span<const uint8_t> shaderSpv;
    vk::PipelineVertexInputStateCreateInfo vertexInfo;
    vk::Format colorFormat;
    vk::Format depthFormat;
};

struct ComputePipelineDesc {
    vk::PipelineLayout layout;
    std::span<const uint8_t> shaderSpv;
};

std::shared_ptr<Pipeline> createGraphicsPipeline(
    const VulkanContext& context,
    const GraphicsPipelineDesc& desc
);

std::shared_ptr<Pipeline> createComputePipeline(
    const VulkanContext& context,
    const ComputePipelineDesc& desc,
    const std::array<uint32_t, 3> numthreads
);

#endif  // RENDERPIPELINE_HPP
