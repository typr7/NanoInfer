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
    CudaDeviceBuffer hidden_state;
    CudaDeviceBuffer kv_cache;
    DeviceArena workspace;
};

std::vector<std::int32_t> inference(
    const std::vector<std::int32_t>& input_tokens,
    const Llama3_2& weights,
    InferenceContext& context
);