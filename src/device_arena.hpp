#pragma once

#include <cstddef>
#include <new>

#include "cuda_device_buffer.h"

class DeviceArena
{
public:
    DeviceArena(const DeviceArena&) = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;

    DeviceArena(std::size_t bytes): capacity(bytes)
    {
        buffer.resize(bytes);
    }

    DeviceArena(DeviceArena&& other) noexcept = default;

    DeviceArena& operator=(DeviceArena&& other) noexcept = default;

    ~DeviceArena() = default;

    void reset()
    {
        offset = 0;
    }

    void* alloc_bytes(std::size_t bytes, std::size_t alignment = 256)
    {
        std::size_t aligned = (offset + alignment - 1) & ~(alignment - 1);

        if (aligned + bytes > capacity) {
            throw std::bad_alloc();
        }

        offset = aligned + bytes;
        void* ptr = buffer.data<std::byte>() + aligned;
        return ptr;
    }

    template <typename T>
    T* alloc(std::size_t size, std::size_t alignment = 256)
    {
        return static_cast<T*>(alloc_bytes(size * sizeof(T), alignment));
    }

private:
    CudaDeviceBuffer buffer;
    std::size_t capacity = 0;
    std::size_t offset = 0;
};
