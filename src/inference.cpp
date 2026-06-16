#include <vector>
#include <cstdint>
#include <algorithm>

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

std::int32_t run_prefill_stage(
    const std::vector<std::int32_t>& input_token_ids,
    const Llama3_2& weights,
    InferenceContext& context
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
    InferenceContext& context
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
    }

    workspace.reset();

    auto* norm_buffer = workspace.alloc<__nv_bfloat16>(HIDDEN_DIM);
    auto* vocab_logits = workspace.alloc<__nv_bfloat16>(VOCAB_LEN);

    launch_rms_norm_kernel(1, hidden_state, weights.norm, norm_buffer, stream);

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
    InferenceContext& context
) {
    const std::size_t input_token_count = token_ids.size();
    if (input_token_count == 0 || input_token_count >= MAX_TOKEN_LEN) {
        return 0;
    }

    std::int32_t token_id = run_prefill_stage(token_ids, weights, context);
    if (is_eot(token_id)) {
        return 0;
    }
    token_ids.push_back(token_id);

    std::size_t token_count = input_token_count + 1;
    for (; token_count < MAX_TOKEN_LEN; token_count++) {
        token_id = run_decode_step(token_count, weights, context);
        if (is_eot(token_id)) {
            break;
        }
        token_ids.push_back(token_id);
    }

    return token_count - input_token_count;
}
