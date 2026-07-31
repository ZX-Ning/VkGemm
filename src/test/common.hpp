#ifndef COMMON_HPP
#define COMMON_HPP

#include <vector>
#include <vulkan/vulkan.hpp>

#include "../core/Buffer.hpp"

inline std::vector<vk::DescriptorSetLayoutBinding> createComputeBindingLayouts(int n) {
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

inline void WriteDescriptorSet(
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

#endif  // COMMON_HPP
