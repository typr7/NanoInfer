#include <vector>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <utility>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include "inference.h"
#include "macro.h"
#include "llama.h"
#include "device_arena.hpp"
#include "kernels/token_embedding.h"
#include "kernels/rms_norm.h"
#include "kernels/greedy_decoding.h"


namespace
{

constexpr std::size_t BF16_BYTES = sizeof(__nv_bfloat16);
constexpr std::size_t WORKSPACE_BYTES = 2048ull << 20;

constexpr std::size_t token_id_buffer_bytes()
{
    return static_cast<std::size_t>(MAX_TOKEN_LEN) * sizeof(std::int32_t);
}

constexpr std::size_t hidden_state_buffer_bytes()
{
    return static_cast<std::size_t>(MAX_TOKEN_LEN) * HIDDEN_DIM * BF16_BYTES;
}

constexpr std::size_t kv_cache_buffer_bytes()
{
    return static_cast<std::size_t>(LAYER_NUM) * MAX_TOKEN_LEN * 2 * K_PROJ_DIM * BF16_BYTES;
}

template <typename T>
float logit_to_float(T value)
{
    return static_cast<float>(value);
}

template <>
float logit_to_float<__nv_bfloat16>(__nv_bfloat16 value)
{
    return static_cast<float>(value);
}

template <typename T>
bool better_host_logit(
    const std::vector<T>& logits,
    std::int32_t lhs,
    std::int32_t rhs
) {
    const float lhs_logit = logit_to_float(logits[lhs]);
    const float rhs_logit = logit_to_float(logits[rhs]);
    return (lhs_logit > rhs_logit) || (lhs_logit == rhs_logit && lhs < rhs);
}

InferenceStepTrace& ensure_step_trace(std::size_t step, InferenceTrace* trace)
{
    auto it = std::find_if(
        trace->steps.begin(),
        trace->steps.end(),
        [&](const InferenceStepTrace& step_trace) {
            return step_trace.step == step;
        }
    );
    if (it != trace->steps.end()) {
        return *it;
    }

    InferenceStepTrace step_trace;
    step_trace.step = step;
    trace->steps.push_back(std::move(step_trace));
    return trace->steps.back();
}

template <typename T>
void append_top_logits(
    const std::vector<T>& host_logits,
    std::size_t step,
    InferenceTrace* trace
) {
    const std::size_t top_k = std::min<std::size_t>(trace->top_k, VOCAB_LEN);
    std::vector<std::int32_t> token_ids(VOCAB_LEN);
    std::iota(token_ids.begin(), token_ids.end(), 0);
    std::partial_sort(
        token_ids.begin(),
        token_ids.begin() + static_cast<std::ptrdiff_t>(top_k),
        token_ids.end(),
        [&](std::int32_t lhs, std::int32_t rhs) {
            return better_host_logit(host_logits, lhs, rhs);
        }
    );

    InferenceStepTrace& step_trace = ensure_step_trace(step, trace);
    step_trace.top_logits.clear();
    step_trace.top_logits.reserve(top_k);
    for (std::size_t i = 0; i < top_k; i++) {
        const std::int32_t token_id = token_ids[i];
        step_trace.top_logits.push_back({
            token_id,
            logit_to_float(host_logits[token_id]),
        });
    }
}

void capture_top_logits(
    const __nv_bfloat16* vocab_logits,
    std::size_t step,
    InferenceTrace* trace,
    cudaStream_t stream
) {
    if (trace == nullptr || trace->top_k == 0) {
        return;
    }

    std::vector<__nv_bfloat16> host_logits(VOCAB_LEN);
    CUDA_CHECK(cudaMemcpyAsync(
        host_logits.data(),
        vocab_logits,
        host_logits.size() * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToHost,
        stream
    ));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    append_top_logits(host_logits, step, trace);
}

void capture_top_logits(
    const float* vocab_logits,
    std::size_t step,
    InferenceTrace* trace,
    cudaStream_t stream
) {
    if (trace == nullptr || trace->top_k == 0) {
        return;
    }

    std::vector<float> host_logits(VOCAB_LEN);
    CUDA_CHECK(cudaMemcpyAsync(
        host_logits.data(),
        vocab_logits,
        host_logits.size() * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream
    ));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    append_top_logits(host_logits, step, trace);
}

void capture_final_norm(
    const __nv_bfloat16* final_norm,
    std::size_t step,
    InferenceTrace* trace,
    cudaStream_t stream
) {
    if (trace == nullptr ||
        !trace->capture_final_norm ||
        trace->final_norm_step != step) {
        return;
    }

    std::vector<__nv_bfloat16> host_norm(HIDDEN_DIM);
    CUDA_CHECK(cudaMemcpyAsync(
        host_norm.data(),
        final_norm,
        host_norm.size() * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToHost,
        stream
    ));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    InferenceStepTrace& step_trace = ensure_step_trace(step, trace);
    step_trace.final_norm.clear();
    step_trace.final_norm.reserve(HIDDEN_DIM);
    for (__nv_bfloat16 value: host_norm) {
        step_trace.final_norm.push_back(static_cast<float>(value));
    }
}

void capture_layer_hidden_state(
    const __nv_bfloat16* hidden_state,
    std::size_t step,
    int layer_idx,
    InferenceTrace* trace,
    cudaStream_t stream
) {
    if (trace == nullptr ||
        !trace->capture_layer_hidden_states ||
        trace->layer_hidden_step != step) {
        return;
    }

    std::vector<__nv_bfloat16> host_hidden(HIDDEN_DIM);
    CUDA_CHECK(cudaMemcpyAsync(
        host_hidden.data(),
        hidden_state,
        host_hidden.size() * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToHost,
        stream
    ));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    InferenceStepTrace& step_trace = ensure_step_trace(step, trace);
    if (step_trace.layer_hidden_states.size() < static_cast<std::size_t>(layer_idx + 1)) {
        step_trace.layer_hidden_states.resize(static_cast<std::size_t>(layer_idx + 1));
    }

    std::vector<float>& layer_hidden = step_trace.layer_hidden_states[layer_idx];
    layer_hidden.clear();
    layer_hidden.reserve(HIDDEN_DIM);
    for (__nv_bfloat16 value: host_hidden) {
        layer_hidden.push_back(static_cast<float>(value));
    }
}

std::int32_t run_prefill_stage(
    const std::vector<std::int32_t>& input_token_ids,
    const Llama3_2& weights,
    InferenceContext& context,
    InferenceTrace* trace
) {
    const std::size_t token_count = input_token_ids.size();

    cudaStream_t stream = context.stream;
    cublasHandle_t handle = context.handle;
    CudaDeviceBuffer& token_ids = context.token_ids;
    CudaDeviceBuffer& next_token_id = context.next_token_id;
    auto* hidden_state = context.hidden_state.data<__nv_bfloat16>();
    DeviceArena& workspace = context.workspace;

    CUBLAS_CHECK(cublasSetStream(handle, stream));

    workspace.reset();

    token_ids.upload(input_token_ids.data(), token_count * sizeof(std::int32_t));
    launch_token_embedding_kernel(
        token_count,
        token_ids.data<std::int32_t>(),
        weights.embed_tokens,
        hidden_state,
        stream
    );

    for (int layer_idx = 0; layer_idx < LAYER_NUM; layer_idx++) {
        run_llama_layer_prefill(layer_idx, token_count, weights, context);
        capture_layer_hidden_state(
            hidden_state + (token_count - 1) * HIDDEN_DIM,
            0,
            layer_idx,
            trace,
            stream
        );
    }

    workspace.reset();

    auto* norm_buffer  = workspace.alloc<__nv_bfloat16>(token_count * HIDDEN_DIM);
    auto* vocab_logits = workspace.alloc<__nv_bfloat16>(VOCAB_LEN);

    launch_rms_norm_kernel(
        token_count,
        hidden_state,
        weights.norm,
        norm_buffer,
        stream
    );
    capture_final_norm(
        norm_buffer + (token_count - 1) * HIDDEN_DIM,
        0,
        trace,
        stream
    );

    CUBLAS_CHECK(
        cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            VOCAB_LEN, 1 /* for single batch */, HIDDEN_DIM,
            &CUBLAS_ALPHA_ONE,
            weights.embed_tokens, CUDA_R_16BF, HIDDEN_DIM,
            norm_buffer + (token_count - 1) * HIDDEN_DIM, CUDA_R_16BF, HIDDEN_DIM,
            &CUBLAS_BETA_ZERO,
            vocab_logits, CUDA_R_16BF, VOCAB_LEN,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        )
    );

    if (trace != nullptr && trace->fp32_logits) {
        auto* fp32_vocab_logits = workspace.alloc<float>(VOCAB_LEN);
        CUBLAS_CHECK(
            cublasGemmEx(
                handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                VOCAB_LEN, 1 /* for single batch */, HIDDEN_DIM,
                &CUBLAS_ALPHA_ONE,
                weights.embed_tokens, CUDA_R_16BF, HIDDEN_DIM,
                norm_buffer + (token_count - 1) * HIDDEN_DIM, CUDA_R_16BF, HIDDEN_DIM,
                &CUBLAS_BETA_ZERO,
                fp32_vocab_logits, CUDA_R_32F, VOCAB_LEN,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT
            )
        );
        capture_top_logits(fp32_vocab_logits, 0, trace, stream);
    } else {
        capture_top_logits(vocab_logits, 0, trace, stream);
    }

    launch_single_batch_greedy_decode_kernel(
        vocab_logits,
        weights.embed_tokens,
        hidden_state,
        next_token_id.data<std::int32_t>(),
        stream
    );

    workspace.reset();

    std::int32_t token_id;
    next_token_id.download_async(&token_id, sizeof(std::int32_t), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return token_id;
}

std::int32_t run_decode_step(
    std::size_t token_count,
    const Llama3_2& weights,
    InferenceContext& context,
    std::size_t step,
    InferenceTrace* trace
) {
    cudaStream_t stream = context.stream;
    cublasHandle_t handle = context.handle;
    DeviceArena& workspace = context.workspace;
    CudaDeviceBuffer& next_token_id = context.next_token_id;
    auto* hidden_state = context.hidden_state.data<__nv_bfloat16>();

    CUBLAS_CHECK(cublasSetStream(handle, stream));

    workspace.reset();

    for (int layer_idx = 0; layer_idx < LAYER_NUM; layer_idx++) {
        run_llama_layer_decode(layer_idx, token_count, weights, context);
        capture_layer_hidden_state(hidden_state, step, layer_idx, trace, stream);
    }

    workspace.reset();

    auto* norm_buffer = workspace.alloc<__nv_bfloat16>(HIDDEN_DIM);
    auto* vocab_logits = workspace.alloc<__nv_bfloat16>(VOCAB_LEN);

    launch_rms_norm_kernel(1, hidden_state, weights.norm, norm_buffer, stream);
    capture_final_norm(norm_buffer, step, trace, stream);

    CUBLAS_CHECK(
        cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            VOCAB_LEN, 1, HIDDEN_DIM,
            &CUBLAS_ALPHA_ONE,
            weights.embed_tokens, CUDA_R_16BF, HIDDEN_DIM,
            norm_buffer, CUDA_R_16BF, HIDDEN_DIM,
            &CUBLAS_BETA_ZERO,
            vocab_logits, CUDA_R_16BF, VOCAB_LEN,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        )
    );

    if (trace != nullptr && trace->fp32_logits) {
        auto* fp32_vocab_logits = workspace.alloc<float>(VOCAB_LEN);
        CUBLAS_CHECK(
            cublasGemmEx(
                handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                VOCAB_LEN, 1, HIDDEN_DIM,
                &CUBLAS_ALPHA_ONE,
                weights.embed_tokens, CUDA_R_16BF, HIDDEN_DIM,
                norm_buffer, CUDA_R_16BF, HIDDEN_DIM,
                &CUBLAS_BETA_ZERO,
                fp32_vocab_logits, CUDA_R_32F, VOCAB_LEN,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT
            )
        );
        capture_top_logits(fp32_vocab_logits, step, trace, stream);
    } else {
        capture_top_logits(vocab_logits, step, trace, stream);
    }

    launch_single_batch_greedy_decode_kernel(
        vocab_logits,
        weights.embed_tokens,
        hidden_state,
        next_token_id.data<std::int32_t>(),
        stream
    );

    workspace.reset();

    std::int32_t token_id;
    next_token_id.download_async(&token_id, sizeof(std::int32_t), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return token_id;
}

bool is_eot(std::int32_t token_id)
{
    return (token_id == 128001 || token_id == 128008 || token_id == 128009);
}

}

InferenceContext::InferenceContext():
    token_ids(token_id_buffer_bytes()),
    next_token_id(sizeof(std::int32_t)),
    hidden_state(hidden_state_buffer_bytes()),
    kv_cache(kv_cache_buffer_bytes()),
    workspace(WORKSPACE_BYTES)
{
    try {
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUBLAS_CHECK(cublasCreate(&handle));
    } catch (...) {
        if (handle != nullptr) {
            cublasDestroy(handle);
            handle = nullptr;
        }
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
        throw;
    }
}

InferenceContext::~InferenceContext() noexcept
{
    if (stream != nullptr) {
        cudaStreamSynchronize(stream);
    }
    if (handle != nullptr) {
        cublasDestroy(handle);
        handle = nullptr;
    }
    if (stream != nullptr) {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }
}

std::size_t inference(
    std::vector<std::int32_t>& token_ids,
    const Llama3_2& weights,
    InferenceContext& context,
    std::size_t max_new_tokens,
    InferenceTrace* trace
) {
    const std::size_t input_token_count = token_ids.size();
    if (input_token_count == 0 || input_token_count >= MAX_TOKEN_LEN || max_new_tokens == 0) {
        return 0;
    }

    if (trace != nullptr) {
        trace->steps.clear();
    }

    std::int32_t token_id = run_prefill_stage(token_ids, weights, context, trace);
    if (is_eot(token_id)) {
        return 0;
    }
    token_ids.push_back(token_id);
    std::size_t generated_count = 1;

    while (generated_count < max_new_tokens && token_ids.size() < MAX_TOKEN_LEN) {
        token_id = run_decode_step(token_ids.size(), weights, context, generated_count, trace);
        if (is_eot(token_id)) {
            break;
        }
        token_ids.push_back(token_id);
        generated_count++;
    }

    return generated_count;
}
