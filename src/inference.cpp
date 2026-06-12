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

void run_prefill(
    const std::vector<std::int32_t>& input_tokens,
    const Llama3_2& weights,
    InferenceContext& context
) {
    const std::size_t token_count = input_tokens.size();

    cudaStream_t stream = context.stream;
    CudaDeviceBuffer& token_ids = context.token_ids;
    auto* hidden_state = context.hidden_state.data<__nv_bfloat16>();
    DeviceArena& workspace = context.workspace;

    workspace.reset();

    token_ids.upload(input_tokens.data(), token_count * sizeof(std::int32_t));
    launch_token_embedding_kernel(
        token_count,
        token_ids.data<std::int32_t>(),
        weights.embed_tokens,
        hidden_state,
        stream
    );

    for (int layer_idx = 0; layer_idx < LAYER_NUM; layer_idx++) {
        run_llama_layer_prefill(weights, layer_idx, token_count, context);
    }

    auto* norm_buffer  = workspace.alloc<__nv_bfloat16>(token_count * HIDDEN_DIM);
    auto* vocab_logits = workspace.alloc<__nv_bfloat16>(VOCAB_LEN);

    launch_rms_norm_kernel(
        token_count,
        RMS_NORM_EPS,
        hidden_state,
        weights.norm,
        norm_buffer,
        stream
    );

    CUBLAS_CHECK(
        cublasGemmEx(
            context.handle,
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

    launch_single_batch_greedy_decode_prefill_kernel(
        vocab_logits,
        weights.embed_tokens,
        hidden_state,
        token_ids.data<std::int32_t>() + token_count,
        stream
    );

    workspace.reset();
}

}

std::vector<std::int32_t> inference(
    const std::vector<std::int32_t>& input_tokens,
    const Llama3_2& weights,
    InferenceContext& context
) {
    const std::size_t token_count = input_tokens.size();
    if (token_count == 0) {
        return {};
    }

    run_prefill(input_tokens, weights, context);

    // TODO: decode stage

    return {}; // dummy
}