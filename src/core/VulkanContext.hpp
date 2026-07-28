#ifndef VULKANCONTEXT_HPP
#define VULKANCONTEXT_HPP

#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "../utils.hpp"
#include "VulkanUtils.hpp"

struct WindowApp;
struct VulkanContext;

struct VulkanContext {
    vk::raii::Context context;
    vk::raii::Instance instance{nullptr};
    vk::raii::DebugUtilsMessengerEXT debugMessenger{nullptr};
    vk::raii::PhysicalDevice physicalDevice{nullptr};
    vk::raii::Device device{nullptr};
    // vk::raii::SurfaceKHR surface{nullptr};
    vk::raii::Queue queue{nullptr};
    vk::raii::CommandPool commandPool{nullptr};
    uint32_t queueFamilyIndex = ~0;
    vk::raii::CommandBuffer loadingCmdBuffer{nullptr};
    vk::raii::DescriptorPool descriptorPool{nullptr};
    uint32_t subgroupSize;

    VmaAllocatorWrapper allocator;
    // vk::SurfaceFormatKHR surfaceFormat;
    void initLogicalDevice();
    void initVmaAllocator();
    explicit VulkanContext();
    ~VulkanContext();

    bool supportsCooperativeMatrix2() const;

    DISABLE_COPY(VulkanContext)
    VulkanContext(VulkanContext&&) = delete;
    VulkanContext& operator=(VulkanContext&&) = delete;

private:
    bool cooperativeMatrix2Supported = false;
};

#endif  // VULKANCONTEXT_HPP
