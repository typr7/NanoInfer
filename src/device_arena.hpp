#pragma once

#include <cstddef>
#include <new>
#include <utility>

#include <cuda_runtime.h>

#include "macro.h"

class DeviceArena
{
public:
    DeviceArena(const DeviceArena&) = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;

    DeviceArena(std::size_t bytes): capacity(bytes)
    {
        CUDA_CHECK(cudaMalloc(&data, bytes));
    }

    DeviceArena(DeviceArena&& other) noexcept
    {
        *this = std::move(other);
    }

    DeviceArena& operator=(DeviceArena&& other) noexcept
    {
        if (this == &other) {
            return *this;
        }

        if (data) {
            cudaFree(data);
        }

        data = other.data;
        capacity = other.capacity;
        offset = other.offset;

        other.data = nullptr;
        other.capacity = other.offset = 0;

        return *this;
    }

    ~DeviceArena()
    {
        if (data) {
            cudaFree(data);
            data = nullptr;
        }
    }

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
        void* ptr = static_cast<std::byte*>(data) + aligned;
        return ptr;
    }

    template <typename T>
    T* alloc(std::size_t size, std::size_t alignment = 256)
    {
        return static_cast<T*>(alloc_bytes(size * sizeof(T), alignment));
    }

private:
    void* data = nullptr;
    std::size_t capacity = 0;
    std::size_t offset = 0;
};
