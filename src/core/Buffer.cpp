#define VMA_DEBUG_INITIALIZE_ALLOCATIONS 1
#include "Buffer.hpp"

LoadedBuffer::LoadedBuffer(
    const vk::BufferCreateInfo& info,
    const VmaAllocationCreateInfo& allocInfo,
    const VmaAllocator& allocator
) : allocator(allocator) {
    VkBuffer buffer;
    VkResult result = vmaCreateBuffer(
        allocator,
        &*info,
        &allocInfo,
        &buffer,
        &allocation,
        &resultInfo
    );
    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::format("Can not create buffer: {}", static_cast<int>(result))
        );
    }
    resultInfo.size = info.size;
    this->buffer = vk::Buffer(buffer);
}

size_t LoadedBuffer::size() const {
    return this->resultInfo.size;
}

vk::Buffer LoadedBuffer::getVkBuffer() const {
    return buffer;
}

LoadedBuffer::~LoadedBuffer() {
    if (buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator, buffer, allocation);
    }
}

DynamicBuffer::DynamicBuffer(
    const vk::BufferCreateInfo& info,
    const VmaAllocationCreateInfo& allocInfo,
    const VmaAllocator& allocator
) : LoadedBuffer(info, allocInfo, allocator) {}

void DynamicBuffer::update(std::span<const uint8_t> data) {
    std::memcpy(resultInfo.pMappedData, data.data(), data.size_bytes());
}

uint8_t* DynamicBuffer::getMappedPtr() {
    return (uint8_t*)this->resultInfo.pMappedData;
}

void StaticBuffer::readBackSyncDangerous(VulkanContext& ctx, uint8_t* dst) {
    size_t stagingSize = std::min(resultInfo.size, MAX_STAGGING_SIZE);
    staging = BufferFactory::createStagingBuffer(allocator, stagingSize);
    for (size_t offset = 0; offset < resultInfo.size; offset += MAX_STAGGING_SIZE) {
        size_t copySize = std::min(MAX_STAGGING_SIZE, resultInfo.size - offset);
        ctx.loadingCmdBuffer.begin({});
        ctx.loadingCmdBuffer.copyBuffer(  // copy device buf to stagging
            this->buffer,
            staging->getVkBuffer(),
            vk::BufferCopy{offset, 0, copySize}
        );
        ctx.loadingCmdBuffer.end();
        ctx.queue.submit({
            vk::SubmitInfo{}.setCommandBuffers({*ctx.loadingCmdBuffer}),
        });
        ctx.device.waitIdle();
        std::memcpy(dst + offset, staging->getMappedPtr(), copySize);
    }
    deleteStaging();
}

StaticBuffer::StaticBuffer(
    const vk::BufferCreateInfo& info,
    const VmaAllocationCreateInfo& allocInfo,
    const VmaAllocator& allocator
) : LoadedBuffer(info, allocInfo, allocator) {}

void StaticBuffer::copyBuffer(vk::Buffer src, vk::Buffer dst, vk::raii::CommandBuffer& cmd, vk::DeviceSize size) {
    cmd.copyBuffer(
        src,
        dst,
        vk::BufferCopy(0, 0, size)
    );
}

void StaticBuffer::load(
    std::span<const uint8_t> data, vk::raii::CommandBuffer& cmd
) {
    int size = resultInfo.size;
    staging = BufferFactory::createStagingBuffer(allocator, size);
    staging->update(data);
    copyBuffer(staging->getVkBuffer(), this->buffer, cmd, size);
}

void StaticBuffer::loadSync(std::span<const uint8_t> data, VulkanContext& ctx) {
    size_t stagingSize = std::min(resultInfo.size, MAX_STAGGING_SIZE);
    staging = BufferFactory::createStagingBuffer(allocator, stagingSize);
    for (size_t offset = 0; offset < data.size_bytes(); offset += MAX_STAGGING_SIZE) {
        size_t copySize = std::min(MAX_STAGGING_SIZE, resultInfo.size - offset);
        auto slice = data.subspan(offset, copySize);
        staging->update(slice);
        ctx.loadingCmdBuffer.begin({});
        ctx.loadingCmdBuffer.copyBuffer(
            staging->getVkBuffer(),
            this->buffer,
            vk::BufferCopy{0, offset, copySize}
        );
        ctx.loadingCmdBuffer.end();
        ctx.queue.submit({
            vk::SubmitInfo{}.setCommandBuffers({*ctx.loadingCmdBuffer}),
        });
        ctx.device.waitIdle();
    }
    deleteStaging();
}

std::span<uint8_t> StaticBuffer::readBackToMapped(vk::raii::CommandBuffer& cmd) {
    size_t size = resultInfo.size;
    staging = BufferFactory::createStagingBuffer(allocator, size);
    copyBuffer(this->buffer, staging->getVkBuffer(), cmd, size);
    return {staging->getMappedPtr(), size};
}

void StaticBuffer::deleteStaging() {
    staging.reset(nullptr);
}

std::unique_ptr<DynamicBuffer> BufferFactory::createStagingBuffer(const VmaAllocator& allocator, size_t size) {
    vk::BufferCreateInfo stagingInfo{
        .size = size,
        .usage = vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    VmaAllocationCreateInfo stagingAllocInfo = {
        .flags =
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
        .requiredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    };
    return std::make_unique<DynamicBuffer>(stagingInfo, stagingAllocInfo, allocator);
}

std::shared_ptr<StaticBuffer> BufferFactory::createStaticBuffer(
    BufferFactory::Type type, const VmaAllocator& allocator, size_t size
) {
    vk::BufferCreateInfo bufferInfo{
        .size = size,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    VmaAllocationCreateInfo allocInfo = {
        .usage = VMA_MEMORY_USAGE_AUTO,
    };
    switch (type) {
        case Type::Vertex: {
            bufferInfo.usage =
                vk::BufferUsageFlagBits::eVertexBuffer |
                vk::BufferUsageFlagBits::eTransferDst;
            allocInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            break;
        }
        case Type::Index: {
            bufferInfo.usage =
                vk::BufferUsageFlagBits::eIndexBuffer |
                vk::BufferUsageFlagBits::eTransferDst;
            allocInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            break;
        }
        case Type::Uniform: {
            bufferInfo.usage =
                vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst;
            allocInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            break;
        }
        case Type::Storage: {
            bufferInfo.usage =
                vk::BufferUsageFlagBits::eStorageBuffer |
                vk::BufferUsageFlagBits::eTransferDst |
                vk::BufferUsageFlagBits::eTransferSrc;
            allocInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            break;
        }
        default: {
            throw std::runtime_error("Not implement yet");
            break;
        }
    }
    return std::make_unique<StaticBuffer>(bufferInfo, allocInfo, allocator);
}

std::shared_ptr<DynamicBuffer> BufferFactory::createDynamicBuffer(
    Type type, const VmaAllocator& allocator, size_t size
) {
    vk::BufferCreateInfo stagingInfo{
        .size = size,
        .usage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    VmaAllocationCreateInfo allocInfo = {
        .flags =
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
        .requiredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    };
    switch (type) {
        case Type::Uniform: {
            return std::make_unique<DynamicBuffer>(stagingInfo, allocInfo, allocator);
        }
        default: {
            throw std::runtime_error("Not implement yet");
            break;
        }
    }
}
