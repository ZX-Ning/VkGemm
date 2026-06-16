#include "RenderPipeline.hpp"

#include <vulkan/vulkan_raii.hpp>

#include "VulkanContext.hpp"

namespace {
vk::raii::ShaderModule createShaderModule(
    const vk::raii::Device& device,
    const std::span<const uint8_t> spv
) {
    vk::ShaderModuleCreateInfo createInfo{
        .codeSize = spv.size() * sizeof(char),
        .pCode = reinterpret_cast<const uint32_t*>(spv.data())
    };
    vk::raii::ShaderModule shaderModule{device, createInfo};

    return shaderModule;
}

}  // namespace

std::shared_ptr<Pipeline> createGraphicsPipeline(
    const VulkanContext& context,
    const GraphicsPipelineDesc& desc
) {
    vk::raii::ShaderModule shaderModule =
        createShaderModule(context.device, desc.shaderSpv);

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eVertex,
        .module = shaderModule,
        .pName = "vertMain"
    };
    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eFragment,
        .module = shaderModule,
        .pName = "fragMain"
    };
    vk::PipelineShaderStageCreateInfo shaderStages[] = {
        vertShaderStageInfo, fragShaderStageInfo
    };

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = vk::PrimitiveTopology::eTriangleList
    };
    vk::PipelineViewportStateCreateInfo viewportState{
        .viewportCount = 1, .scissorCount = 1
    };

    vk::PipelineRasterizationStateCreateInfo rasterizer{
        .depthClampEnable = vk::False,
        .rasterizerDiscardEnable = vk::False,
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eBack,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .depthBiasEnable = vk::False,
        .depthBiasSlopeFactor = 1.0f,
        .lineWidth = 1.0f
    };

    vk::PipelineMultisampleStateCreateInfo multisampling{
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
        .sampleShadingEnable = vk::False
    };

    vk::PipelineColorBlendAttachmentState colorBlendAttachment{
        .blendEnable = vk::False,
        .colorWriteMask =
            vk::ColorComponentFlagBits::eR |
            vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB |
            vk::ColorComponentFlagBits::eA
    };

    vk::PipelineColorBlendStateCreateInfo colorBlending{
        .logicOpEnable = vk::False,
        .logicOp = vk::LogicOp::eCopy,
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachment
    };

    vk::PipelineDepthStencilStateCreateInfo depthStencil{
        .depthTestEnable = vk::True,
        .depthWriteEnable = vk::True,
        .depthCompareOp = vk::CompareOp::eLess,
        .depthBoundsTestEnable = vk::False,
        .stencilTestEnable = vk::False
    };

    std::vector dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dynamicState{
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data()
    };
    vk::StructureChain pipelineCreateInfoChain{
        vk::GraphicsPipelineCreateInfo{
            .stageCount = 2,
            .pStages = shaderStages,
            .pVertexInputState = &desc.vertexInfo,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = &depthStencil,
            .pColorBlendState = &colorBlending,
            .pDynamicState = &dynamicState,
            .layout = desc.layout,
            .renderPass = nullptr,
        },
        vk::PipelineRenderingCreateInfo{
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &desc.colorFormat,
            .depthAttachmentFormat = desc.depthFormat
        }
    };
    return std::make_unique<Pipeline>(vk::raii::Pipeline{
        context.device,
        nullptr,
        pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>()
    });
}

std::shared_ptr<Pipeline> createComputePipeline(
    const VulkanContext& context,
    const ComputePipelineDesc& desc
) {
    vk::raii::ShaderModule shaderModule =
        createShaderModule(context.device, desc.shaderSpv);
    vk::ComputePipelineCreateInfo compPipeInfo = {
        .stage = vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eCompute,
            .module = shaderModule,
            .pName = "main"
        },
        .layout = desc.layout
    };
    return std::make_unique<Pipeline>(
        context.device.createComputePipeline(nullptr, compPipeInfo)
    );
}
