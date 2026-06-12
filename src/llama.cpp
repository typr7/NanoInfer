#include <cmath>

#include <cublas_v2.h>

#include "llama.h"
#include "inference.h"
#include "macro.h"
#include "device_arena.hpp"
#include "kernels/attention_softmax.h"
#include "kernels/residual_connection.h"
#include "kernels/rms_norm.h"
#include "kernels/rope.h"
#include "kernels/swiglu.h"
#include "kernels/kv_cache.h"


void run_llama_layer_prefill(
    const Llama3_2& weights,
    int layer_idx,
    std::size_t token_count,
    InferenceContext& context
) {
    cudaStream_t stream = context.stream;
    cublasHandle_t handle = context.handle;
    DeviceArena& workspace = context.workspace;
    auto* hidden_state = context.hidden_state.data<__nv_bfloat16>();

    CUBLAS_CHECK(cublasSetStream(handle, stream));

    workspace.reset();

    auto* norm_buffer = workspace.alloc<__nv_bfloat16>(token_count * HIDDEN_DIM);
    auto* qkv_buffer = workspace.alloc<__nv_bfloat16>(token_count * (Q_PROJ_DIM + 2 * K_PROJ_DIM));
    auto* attention_scores = workspace.alloc<__nv_bfloat16>(Q_HEAD_NUM * token_count * token_count);
    auto* attention_output = workspace.alloc<__nv_bfloat16>(Q_HEAD_NUM * token_count * HEAD_DIM);

    // rmsnorm
    launch_rms_norm_kernel(
        token_count,
        RMS_NORM_EPS,
        hidden_state,
        weights.input_layernorm[layer_idx],
        norm_buffer,
        stream
    );

    // qkv proj
    const int token_rows = static_cast<int>(token_count);
    const int qkv_stride = Q_PROJ_DIM + 2 * K_PROJ_DIM;
    const int hidden_stride = HIDDEN_DIM;
    CUBLAS_CHECK(
        cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            qkv_stride, token_rows, hidden_stride,
            &CUBLAS_ALPHA_ONE,
            weights.qkv_proj[layer_idx], CUDA_R_16BF, hidden_stride,
            norm_buffer, CUDA_R_16BF, hidden_stride,
            &CUBLAS_BETA_ZERO,
            qkv_buffer, CUDA_R_16BF, qkv_stride,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        )
    );
    
    // qkv: [token_count, query_dim + key_dim + value_dim]

    __nv_bfloat16* query_begin = qkv_buffer;
    __nv_bfloat16* key_begin = query_begin + Q_PROJ_DIM;
    const __nv_bfloat16* value_begin = key_begin + K_PROJ_DIM;

    launch_rope_kernel(token_count, qkv_stride, qkv_stride, query_begin, key_begin, stream);

    launch_single_batch_store_kv_cache_prefill_kernel(
        token_count,
        qkv_stride,
        layer_idx,
        key_begin,
        context.kv_cache.data<__nv_bfloat16>(),
        stream
    );

    const float rsqrt_head_dim = 1.f / std::sqrt(static_cast<float>(HEAD_DIM));

    for (int q_head_idx = 0; q_head_idx < Q_HEAD_NUM; q_head_idx++) {
        const int k_head_idx = q_head_idx / (Q_HEAD_NUM / K_HEAD_NUM);
        const __nv_bfloat16* query_head = query_begin + q_head_idx * HEAD_DIM;
        const __nv_bfloat16* key_head = key_begin + k_head_idx * HEAD_DIM;
        __nv_bfloat16* head_scores = attention_scores + q_head_idx * token_count * token_count;
        CUBLAS_CHECK(
            cublasGemmEx(
                handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                token_rows, token_rows, HEAD_DIM,
                &rsqrt_head_dim,
                key_head, CUDA_R_16BF, qkv_stride,
                query_head, CUDA_R_16BF, qkv_stride,
                &CUBLAS_BETA_ZERO,
                head_scores, CUDA_R_16BF, token_rows,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT
            )
        );
    }

    launch_attention_softmax_prefill_kernel(Q_HEAD_NUM, token_count, attention_scores, stream);

    for (int q_head_idx = 0; q_head_idx < Q_HEAD_NUM; q_head_idx++) {
        const int v_head_idx = q_head_idx / (Q_HEAD_NUM / K_HEAD_NUM);
        const __nv_bfloat16* head_scores = attention_scores + q_head_idx * token_count * token_count;
        const __nv_bfloat16* value_head = value_begin + v_head_idx * HEAD_DIM;
        __nv_bfloat16* output_head = attention_output + q_head_idx * HEAD_DIM;
        CUBLAS_CHECK(
            cublasGemmEx(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                HEAD_DIM, token_rows, token_rows,
                &CUBLAS_ALPHA_ONE,
                value_head, CUDA_R_16BF, qkv_stride,
                head_scores, CUDA_R_16BF, token_rows,
                &CUBLAS_BETA_ZERO,
                output_head, CUDA_R_16BF, Q_PROJ_DIM,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT
            )
        );
    }

    CUBLAS_CHECK(
        cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            HIDDEN_DIM, token_rows, Q_PROJ_DIM,
            &CUBLAS_ALPHA_ONE,
            weights.o_proj[layer_idx], CUDA_R_16BF, Q_PROJ_DIM,
            attention_output, CUDA_R_16BF, Q_PROJ_DIM,
            &CUBLAS_BETA_ZERO,
            norm_buffer, CUDA_R_16BF, HIDDEN_DIM,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        )
    );

    launch_residual_connection_kernel(token_count, norm_buffer, hidden_state, stream);

    workspace.reset();

    const int packed_dim = 2 * UP_PROJ_DIM;
    norm_buffer = workspace.alloc<__nv_bfloat16>(token_count * HIDDEN_DIM);
    auto* gate_up_buffer = workspace.alloc<__nv_bfloat16>(token_count * packed_dim);

    launch_rms_norm_kernel(
        token_count,
        RMS_NORM_EPS,
        hidden_state,
        weights.post_attention_layernorm[layer_idx],
        norm_buffer,
        stream
    );

    CUBLAS_CHECK(
        cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            packed_dim, token_rows, HIDDEN_DIM,
            &CUBLAS_ALPHA_ONE,
            weights.gate_up_proj[layer_idx], CUDA_R_16BF, HIDDEN_DIM,
            norm_buffer, CUDA_R_16BF, HIDDEN_DIM,
            &CUBLAS_BETA_ZERO,
            gate_up_buffer, CUDA_R_16BF, packed_dim,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        )
    );

    launch_swiglu_inplace_kernel(token_count, UP_PROJ_DIM, packed_dim, gate_up_buffer, stream);

    CUBLAS_CHECK(
        cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            HIDDEN_DIM, token_rows, UP_PROJ_DIM,
            &CUBLAS_ALPHA_ONE,
            weights.down_proj[layer_idx], CUDA_R_16BF, UP_PROJ_DIM,
            gate_up_buffer, CUDA_R_16BF, packed_dim,
            &CUBLAS_BETA_ZERO,
            norm_buffer, CUDA_R_16BF, HIDDEN_DIM,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        )
    );

    launch_residual_connection_kernel(token_count, norm_buffer, hidden_state, stream);

    workspace.reset();
}
