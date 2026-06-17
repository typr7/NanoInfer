#pragma once


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

struct LogitTopKEntry
{
    std::int32_t token_id = 0;
    float logit = 0.f;
};

struct InferenceStepTrace
{
    std::size_t step = 0;
    std::vector<LogitTopKEntry> top_logits;
    std::vector<float> final_norm;
    std::vector<std::vector<float>> layer_hidden_states;
};

struct InferenceTrace
{
    std::size_t top_k = 0;
    bool fp32_logits = false;
    bool capture_final_norm = false;
    std::size_t final_norm_step = 0;
    bool capture_layer_hidden_states = false;
    std::size_t layer_hidden_step = 0;
    std::vector<InferenceStepTrace> steps;
};

std::size_t inference(
    std::vector<std::int32_t>& token_ids,
    const Llama3_2& weights,
    InferenceContext& context,
    std::size_t max_new_tokens = MAX_TOKEN_LEN,
    InferenceTrace* trace = nullptr
);
