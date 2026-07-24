#include "VulkanContext.hpp"

// #include <SDL3/SDL_error.h>
// #include <SDL3/SDL_vulkan.h>
#include <vk_mem_alloc.h>

#include <cassert>
#include <format>
#include <print>
#include <stdexcept>

// vulkan
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "../utils.hpp"

namespace {

constexpr bool ENABLE_VALIDATION_LAYERS = !IS_RELEASE;
const std::vector<char const*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};
const std::vector<const char*> requiredDeviceExtensions = {
    // vk::KHRSwapchainExtensionName,
    vk::KHRSpirv14ExtensionName,
    vk::KHRSynchronization2ExtensionName,
    vk::KHRCreateRenderpass2ExtensionName,
    vk::KHRDynamicRenderingExtensionName,
    vk::EXTMemoryBudgetExtensionName,
    vk::KHRCooperativeMatrixExtensionName,
    vk::KHRMaintenance4ExtensionName
};

constexpr uint32_t COOPERATIVE_MATRIX_TILE_SIZE = 16;
// constexpr uint32_t COOPERATIVE_MATRIX_SUBGROUP_SIZE = 32;

bool supportsRequiredCooperativeMatrixType(
    const vk::raii::PhysicalDevice& device
) {
    const auto properties = device.getCooperativeMatrixPropertiesKHR();
    for (const auto& property : properties) {
        if (property.MSize == COOPERATIVE_MATRIX_TILE_SIZE &&
            property.NSize == COOPERATIVE_MATRIX_TILE_SIZE &&
            property.KSize == COOPERATIVE_MATRIX_TILE_SIZE &&
            property.AType == vk::ComponentTypeKHR::eFloat16 &&
            property.BType == vk::ComponentTypeKHR::eFloat16 &&
            property.CType == vk::ComponentTypeKHR::eFloat32 &&
            property.ResultType == vk::ComponentTypeKHR::eFloat32 &&
            property.scope == vk::ScopeKHR::eSubgroup) {
            return true;
        }
    }
    return false;
}

std::vector<const char*> getRequiredExtensions() {
    // Uint32 sdlExtensionCount = 0;
    // const char* const* sdlExtensions =
    //     SDL_Vulkan_GetInstanceExtensions(&sdlExtensionCount);
    // if (!sdlExtensions) {
    //     throw std::runtime_error(
    //         std::format("failed to get SDL Vulkan instance extensions: {}", SDL_GetError())
    //     );
    // }

    std::vector<const char*> extensions;
    if (ENABLE_VALIDATION_LAYERS) {
        extensions.push_back(vk::EXTDebugUtilsExtensionName);
    }
    return extensions;
}

// vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
//     const std::vector<vk::SurfaceFormatKHR>& availableFormats
// ) {
//     assert(!availableFormats.empty());
//     for (const auto& format : availableFormats) {
//         if ((format.format == vk::Format::eB8G8R8A8Srgb ||
//              format.format == vk::Format::eR8G8B8A8Srgb) &&
//             format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
//             return format;
//         }
//     }
//     for (const auto& format : availableFormats) {
//         if ((format.format == vk::Format::eB8G8R8A8Unorm ||
//              format.format == vk::Format::eR8G8B8A8Unorm) &&
//             format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
//             std::println("Can not found Srgb Format, using Unorm.");
//             return format;
//         }
//     }
//     throw std::runtime_error("Format not supported yet");
//     // return availableFormats[0];
// }

vk::raii::Instance createInstance(const vk::raii::Context& context) {
    constexpr vk::ApplicationInfo appInfo = {
        .pApplicationName = "Learn Vulkan",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = vk::ApiVersion13
    };

    // Get the required layers
    std::vector<char const*> requiredLayers;
    if (ENABLE_VALIDATION_LAYERS) {
        std::println("Enable validation layers.");
        requiredLayers.assign(validationLayers.begin(), validationLayers.end());
    }
    else {
        std::println("Release mode. Disable validation layers.");
    }

    // Check if the required layers are supported by the Vulkan implementation.
    auto layerProperties = context.enumerateInstanceLayerProperties();
    for (auto const& requiredLayer : requiredLayers) {
        bool layerFound = false;
        for (auto const& layerProperty : layerProperties) {
            if (strcmp(layerProperty.layerName, requiredLayer) == 0) {
                layerFound = true;
                break;
            }
        }

        if (!layerFound) {
            throw std::runtime_error(
                std::format("Required layer not supported: {}", std::string(requiredLayer))
            );
        }
    }

    // Get the required extensions.
    auto requiredExtensions = getRequiredExtensions();

    // Check if the required extensions are supported by the Vulkan
    // implementation.
    auto extensionProperties =
        context.enumerateInstanceExtensionProperties();
    for (auto const& requiredExtension : requiredExtensions) {
        bool extensionFound = false;
        for (auto const& extensionProperty : extensionProperties) {
            if (strcmp(extensionProperty.extensionName, requiredExtension) == 0) {
                extensionFound = true;
                break;
            }
        }

        if (!extensionFound) {
            throw std::runtime_error(
                std::format("Required extension not supported: {}", requiredExtension)
            );
        }
    }

    vk::InstanceCreateInfo createInfo{
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
        .ppEnabledLayerNames = requiredLayers.data(),
        .enabledExtensionCount =
            static_cast<uint32_t>(requiredExtensions.size()),
        .ppEnabledExtensionNames = requiredExtensions.data()
    };
    return {context, createInfo};
}

vk::raii::DebugUtilsMessengerEXT setupDebugMessenger(const vk::raii::Instance& instance) {
    if (!ENABLE_VALIDATION_LAYERS) {
        return nullptr;
    }
    vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
    );
    vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(
        vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
        vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
        vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
    );
    vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{
        .messageSeverity = severityFlags,
        .messageType = messageTypeFlags,
        .pfnUserCallback =
            [](
                vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
                vk::DebugUtilsMessageTypeFlagsEXT type,
                const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
                void*
            ) -> vk::Bool32 {
            if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError ||
                severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
                std::println(
                    "[{}] {}, {}",
                    vk::to_string(severity),
                    vk::to_string(type),
                    pCallbackData->pMessage
                );
            }
            return vk::False;
        }
    };
    return instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
}

vk::raii::PhysicalDevice pickPhysicalDevice(vk::raii::Instance& instance) {
    std::vector<vk::raii::PhysicalDevice> devices =
        instance.enumeratePhysicalDevices();

    // for (size_t i = 0; i < devices.size(); i++) {
    //     const auto& d = devices[i];
    //     const auto props = d.getProperties2<
    //         vk::PhysicalDeviceProperties2,
    //         vk::PhysicalDeviceSubgroupProperties,
    //         vk::PhysicalDeviceSubgroupSizeControlProperties>();
    //     uint32_t subgroupSize = props.get<vk::PhysicalDeviceSubgroupProperties>().subgroupSize;
    //     auto subgroupctrlProps = props.get<vk::PhysicalDeviceSubgroupSizeControlProperties>();
    //     auto props2 = props.get<vk::PhysicalDeviceProperties2>().properties;
    //     std::println(
    //         "Device {}: {}, default subgroupSize={} (min={}, max={})",
    //         i,
    //         props2.deviceName.data(),
    //         subgroupSize,
    //         subgroupctrlProps.minSubgroupSize,
    //         subgroupctrlProps.maxSubgroupSize
    //     );
    // }

    // vk::raii::PhysicalDevice* selectedDevice = nullptr;
    std::vector<Ref<vk::raii::PhysicalDevice>> devicesFiltered;
    for (vk::raii::PhysicalDevice& device : devices) {
        // Check if the device supports the Vulkan 1.3 API version
        bool supportsVulkan1_3 =
            device.getProperties().apiVersion >= VK_API_VERSION_1_3;

        // Check if any queue family supports compute operations.
        auto queueFamilies = device.getQueueFamilyProperties();
        bool supported = false;
        for (const vk::QueueFamilyProperties& qfp : queueFamilies) {
            if (qfp.queueFlags & vk::QueueFlagBits::eCompute) {
                supported = true;
                break;
            }
        }

        // Check if all required device extensions are available
        std::vector<vk::ExtensionProperties> availableDeviceExtensions =
            device.enumerateDeviceExtensionProperties();
        bool supportsAllRequiredExtensions = true;
        for (const char* requiredExtension : requiredDeviceExtensions) {
            bool extensionFound = false;
            for (auto const& availableDeviceExtension : availableDeviceExtensions) {
                if (strcmp(availableDeviceExtension.extensionName, requiredExtension) == 0) {
                    extensionFound = true;
                    break;
                }
            }
            if (!extensionFound) {
                supportsAllRequiredExtensions = false;
                break;
            }
        }

        if (!supportsVulkan1_3 || !supported ||
            !supportsAllRequiredExtensions) {
            continue;
        }

        auto features = device.getFeatures2<
            vk::PhysicalDeviceFeatures2,
            vk::PhysicalDeviceVulkan11Features,
            vk::PhysicalDeviceVulkan12Features,
            vk::PhysicalDeviceVulkan13Features,
            vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
            vk::PhysicalDeviceCooperativeMatrixFeaturesKHR>();

        bool supportsRequiredFeatures =
            features.get<vk::PhysicalDeviceVulkan11Features>().storageBuffer16BitAccess &&
            features.get<vk::PhysicalDeviceVulkan11Features>().uniformAndStorageBuffer16BitAccess &&
            features.get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters &&
            features.get<vk::PhysicalDeviceVulkan12Features>().shaderFloat16 &&
            features.get<vk::PhysicalDeviceVulkan12Features>().vulkanMemoryModelDeviceScope &&
            features.get<vk::PhysicalDeviceVulkan12Features>().vulkanMemoryModel &&
            features.get<vk::PhysicalDeviceVulkan13Features>().synchronization2 &&
            features.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
            features.get<vk::PhysicalDeviceVulkan13Features>().maintenance4 &&
            features.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState &&
            features.get<vk::PhysicalDeviceCooperativeMatrixFeaturesKHR>().cooperativeMatrix;

        if (supportsRequiredFeatures &&
            supportsRequiredCooperativeMatrixType(device)) {
            devicesFiltered.push_back(device);
        }
    }
    std::println("Support device count: {}", devicesFiltered.size());
    for (vk::raii::PhysicalDevice& device : devicesFiltered) {
        auto props = device.getProperties();
        if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
            return device;
        }
    }
    if (!devicesFiltered.empty()) {
        auto& device = devicesFiltered.front().get();
        return device;
    }
    throw std::runtime_error(
        "failed to find a GPU supporting 16x16x16 fp16 cooperative matrix GEMM"
    );
}

vk::raii::DescriptorPool createDescriptorPool(
    const vk::raii::Device& device
) {
    std::vector<vk::DescriptorPoolSize> poolSizes{};
    for (int i = 0; i <= 10; i++) {
        poolSizes.push_back({(vk::DescriptorType)i, 1 << 10});
    }
    vk::DescriptorPoolCreateInfo poolInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = static_cast<uint32_t>(1 << 12),
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data()
    };
    return vk::raii::DescriptorPool(device, poolInfo);
}

}  // namespace

VulkanContext::VulkanContext() {
    std::println("Starting Vulkan instance.");

    this->instance = createInstance(context);
    this->debugMessenger = setupDebugMessenger(instance);
    // this->surface = windowApp.createSurface(instance);
    this->physicalDevice = pickPhysicalDevice(instance);
    auto propsChain = this->physicalDevice.getProperties2<
        vk::PhysicalDeviceProperties2,
        vk::PhysicalDeviceSubgroupProperties>();
    auto props2 = propsChain.get<vk::PhysicalDeviceProperties2>();
    this->subgroupSize = propsChain.get<vk::PhysicalDeviceSubgroupProperties>().subgroupSize;
    std::println("Device: {}, subgroupSize={}", props2.properties.deviceName.data(), this->subgroupSize);

    initLogicalDevice();
    initVmaAllocator();
    // Create command pool
    vk::CommandPoolCreateInfo poolInfo{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = queueFamilyIndex
    };
    this->commandPool = vk::raii::CommandPool(device, poolInfo);
    vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1
    };
    this->loadingCmdBuffer =
        std::move(device.allocateCommandBuffers(allocInfo)[0]);
    this->descriptorPool = createDescriptorPool(device);
}

VulkanContext::~VulkanContext() {
    std::println("Cleaning up Vulkan instance.");
};

void VulkanContext::initLogicalDevice() {
    std::vector<vk::QueueFamilyProperties> queueFamilyProperties =
        physicalDevice.getQueueFamilyProperties();

    // Get the first queue family that supports compute operations.
    uint32_t queueIndex = ~0;
    for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++) {
        if (queueFamilyProperties[qfpIndex].queueFlags &
            vk::QueueFlagBits::eCompute) {
            queueIndex = qfpIndex;
            break;
        }
    }
    if (queueIndex == (uint32_t)~0) {
        throw std::runtime_error(
            "Could not find a compute queue -> terminating"
        );
    }

    // query for Vulkan 1.3 features
    vk::StructureChain featureChain{
        vk::PhysicalDeviceFeatures2{},
        vk::PhysicalDeviceVulkan11Features{
            .storageBuffer16BitAccess = true,
            .uniformAndStorageBuffer16BitAccess = true,
            .shaderDrawParameters = true
        },
        vk::PhysicalDeviceVulkan12Features{
            .shaderFloat16 = true,
            .vulkanMemoryModel = true,
            .vulkanMemoryModelDeviceScope = true
        },
        vk::PhysicalDeviceVulkan13Features{
            .synchronization2 = true,
            .dynamicRendering = true,
            .maintenance4 = true
        },
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT{.extendedDynamicState = true},
        vk::PhysicalDeviceCooperativeMatrixFeaturesKHR{.cooperativeMatrix = true}
    };

    // create a Device
    float queuePriority = 0.5f;
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo{
        .queueFamilyIndex = queueIndex,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority
    };
    vk::DeviceCreateInfo deviceCreateInfo{
        .pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &deviceQueueCreateInfo,
        .enabledExtensionCount =
            static_cast<uint32_t>(requiredDeviceExtensions.size()),
        .ppEnabledExtensionNames = requiredDeviceExtensions.data()
    };

    this->device = vk::raii::Device(physicalDevice, deviceCreateInfo);
    this->queueFamilyIndex = queueIndex;
    this->queue = vk::raii::Queue(device, queueFamilyIndex, 0);
}

void VulkanContext::initVmaAllocator() {
    VmaVulkanFunctions functions = {
        .vkGetInstanceProcAddr =
            instance.getDispatcher()
                ->vkGetInstanceProcAddr,
        .vkGetDeviceProcAddr =
            device.getDispatcher()
                ->vkGetDeviceProcAddr,
    };
    VmaAllocatorCreateFlags flags =
        VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
    VmaAllocatorCreateInfo info{
        .flags = flags,
        .physicalDevice = *physicalDevice,
        .device = *device,
        .pVulkanFunctions = &functions,
        .instance = *instance,
        .vulkanApiVersion = VK_API_VERSION_1_3,
    };
    this->allocator = VmaAllocatorWrapper(new VmaAllocator());
    if (vmaCreateAllocator(&info, allocator.get()) != VK_SUCCESS) {
        throw std::runtime_error("Error creating VMA allocator");
    }
}
