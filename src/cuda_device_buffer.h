#pragma once

#include <cstddef>

#include <cuda_runtime.h>


class CudaDeviceBuffer
{
public:
    CudaDeviceBuffer();
    explicit CudaDeviceBuffer(std::size_t byte_size);
    ~CudaDeviceBuffer() noexcept;

    CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
    CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;

    CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept;
    CudaDeviceBuffer& operator=(CudaDeviceBuffer&& other) noexcept;

    template <typename T>
    T* data() noexcept
    {
        return static_cast<T*>(device_data);
    }

    template <typename T>
    const T* data() const noexcept
    {
        return static_cast<const T*>(device_data);
    }

    std::size_t size() const noexcept;
    bool empty() const noexcept;

    void resize(std::size_t byte_size);
    void upload(const void* src, std::size_t byte_size);
    void upload_async(const void* src, std::size_t byte_size, cudaStream_t stream);
    std::size_t download(void* dst, std::size_t byte_size) const;
    void destroy();

private:
    void* device_data = nullptr;
    std::size_t byte_size = 0;
};
