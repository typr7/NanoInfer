#pragma once


#include <vector>
#include <cstdint>

#include <cublas_v2.h>

#include "llama.h"
#include "cuda_device_buffer.h"


struct InferenceContext
{
    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;

    CudaDeviceBuffer token_ids;
    CudaDeviceBuffer next_token_id;
    CudaDeviceBuffer hidden_state;
    CudaDeviceBuffer kv_cache;
    DeviceArena workspace;
};

std::size_t inference(
    std::vector<std::int32_t>& token_ids,
    const Llama3_2& weights,
    InferenceContext& context
);