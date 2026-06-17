#pragma once


#include <cstddef>
#include <vector>
#include <cstdint>

#include <cublas_v2.h>

#include "llama.h"
#include "cuda_device_buffer.h"


struct InferenceContext
{
    InferenceContext();
    ~InferenceContext() noexcept;

    InferenceContext(const InferenceContext&) = delete;
    InferenceContext& operator=(const InferenceContext&) = delete;
    InferenceContext(InferenceContext&&) = delete;
    InferenceContext& operator=(InferenceContext&&) = delete;

    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;

    CudaDeviceBuffer token_ids;
    CudaDeviceBuffer next_token_id;
    CudaDeviceBuffer hidden_state;
    CudaDeviceBuffer kv_cache;
    DeviceArena workspace;
};

struct InferenceBenchmarkResult
{
    std::size_t prompt_tokens = 0;
    std::size_t generated_tokens = 0;
    std::size_t decode_tokens = 0;
    double ttft_ms = 0.0;
    double tpot_ms = 0.0;
    double decode_ms_total = 0.0;
    bool stopped_eos = false;
};

std::size_t inference(
    std::vector<std::int32_t>& token_ids,
    const Llama3_2& weights,
    InferenceContext& context
);

InferenceBenchmarkResult inference_benchmark(
    std::vector<std::int32_t>& token_ids,
    const Llama3_2& weights,
    InferenceContext& context,
    std::size_t max_new_tokens
);
