#ifndef BUFFER_HPP
#define BUFFER_HPP

// vma
#include <vk_mem_alloc.h>

// c++
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

// vulkan
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "../utils.hpp"
#include "VulkanContext.hpp"

constexpr size_t MAX_STAGGING_SIZE = 256 << 20; // 256 MB

struct LoadedBuffer {
protected:
    vk::Buffer buffer;
    VmaAllocation allocation;
    const VmaAllocator& allocator;
    VmaAllocationInfo resultInfo;
    LoadedBuffer(
        const vk::BufferCreateInfo& info,
        const VmaAllocationCreateInfo& allocInfo,
        const VmaAllocator& allocator
    );

public:
    vk::Buffer getVkBuffer() const;
    size_t size() const;
    LoadedBuffer() = delete;
    // rule o 5:
    virtual ~LoadedBuffer();
    DISABLE_COPY(LoadedBuffer)
    LoadedBuffer(LoadedBuffer&&) = delete;
    LoadedBuffer& operator=(LoadedBuffer&&) = delete;
};

struct DynamicBuffer : public LoadedBuffer {
public:
    DynamicBuffer(
        const vk::BufferCreateInfo& info,
        const VmaAllocationCreateInfo& allocInfo,
        const VmaAllocator& allocator
    );

    void update(std::span<const uint8_t> data);
    uint8_t* getMappedPtr();
};

struct StaticBuffer : public LoadedBuffer {
private:
    std::unique_ptr<DynamicBuffer> staging;
    void readBackSyncDangerous(VulkanContext& ctx, uint8_t* dst);
    // size_t size;

public:
    StaticBuffer(
        const vk::BufferCreateInfo& info,
        const VmaAllocationCreateInfo& allocInfo,
        const VmaAllocator& allocator
    );
    static void copyBuffer(vk::Buffer src, vk::Buffer dst, vk::raii::CommandBuffer& cmd, vk::DeviceSize size);
    void load(std::span<const uint8_t> data, vk::raii::CommandBuffer& cmd);
    void loadSync(std::span<const uint8_t> data, VulkanContext& ctx);
    std::span<uint8_t> readBackToMapped(vk::raii::CommandBuffer& cmd);
    template <class T>
    std::vector<T> readBackSync(VulkanContext& ctx);
    void deleteStaging();
};

struct BufferFactory {
    enum class Type {
        Vertex,
        Index,
        Uniform,
        Storage
    };
    static std::shared_ptr<StaticBuffer> createStaticBuffer(Type type, const VmaAllocator& allocator, size_t size);
    static std::shared_ptr<DynamicBuffer> createDynamicBuffer(Type type, const VmaAllocator& allocator, size_t size);
    static std::unique_ptr<DynamicBuffer> createStagingBuffer(const VmaAllocator& allocator, size_t size);
};

template <class T>
inline std::vector<T> StaticBuffer::readBackSync(VulkanContext& ctx) {
    assert(this->resultInfo.size % sizeof(T) == 0);
    std::vector<T> result(this->resultInfo.size / sizeof(T));
    readBackSyncDangerous(ctx, reinterpret_cast<uint8_t*>(result.data()));
    return result;
}

#endif  // BUFFER_HPP
