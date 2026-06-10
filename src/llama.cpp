#include <cmath>

#include <cublas_v2.h>

#include "llama.h"
#include "macro.h"
#include "device_arena.hpp"
#include "kernels.h"
#include "softmax.h"
#include "residual_connection.h"
#include "rope.h"
#include "swiglu.h"


void run_llama32_layer_prefill(
    DeviceArena& arena,
    std::size_t token_num,
    int layer_idx,
    const Llama3_2& llama_weight,
    __nv_bfloat16* hidden_state,
    cudaStream_t stream,
    cublasHandle_t handle
) {
    constexpr float eps  = 0.00001f;
    constexpr float one  = 1.0f;
    constexpr float zero = 0.0f;

    CUBLAS_CHECK(cublasSetStream(handle, stream));

    arena.reset();

    auto* x_norm_buffer = arena.alloc<__nv_bfloat16>(token_num * HIDDEN_DIM);
    auto* qkv_buffer = arena.alloc<__nv_bfloat16>(token_num * (Q_PROJ_DIM + 2 * K_PROJ_DIM));
    auto* attn_prob_buffer = arena.alloc<__nv_bfloat16>(Q_HEAD_NUM * token_num * token_num);
    auto* attn_output_buffer = arena.alloc<__nv_bfloat16>(Q_HEAD_NUM * token_num * HEAD_DIM);

    // rmsnorm
    launchRMSNormKernel(token_num, eps, hidden_state, llama_weight.input_layernorm[layer_idx], x_norm_buffer, stream);

    // qkv proj
    int m = token_num;
    int n = Q_PROJ_DIM + 2 * K_PROJ_DIM;
    int k = HIDDEN_DIM;
    CUBLAS_CHECK(
        cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            n, m, k,
            &one,
            llama_weight.qkv_proj[layer_idx], CUDA_R_16BF, k,
            x_norm_buffer, CUDA_R_16BF, k,
            &zero,
            qkv_buffer, CUDA_R_16BF, n,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        )
    );
    
    // qkv: [token_num, q_dim + k_dim + v_dim]

    __nv_bfloat16* q_begin = qkv_buffer;
    __nv_bfloat16* k_begin = q_begin + Q_PROJ_DIM;
    const __nv_bfloat16* v_begin = k_begin + K_PROJ_DIM;

    launchRoPEKernel(token_num, n, n, q_begin, k_begin, stream);

    const float rsqrt_head_dim = 1.f / std::sqrt(static_cast<float>(HEAD_DIM));

    for (int q_head_idx = 0; q_head_idx < Q_HEAD_NUM; q_head_idx++) {
        const int k_head_idx = q_head_idx / (Q_HEAD_NUM / K_HEAD_NUM);
        const __nv_bfloat16* q = q_begin + q_head_idx * HEAD_DIM;
        const __nv_bfloat16* k = k_begin + k_head_idx * HEAD_DIM;
        __nv_bfloat16* prob = attn_prob_buffer + q_head_idx * token_num * token_num;
        CUBLAS_CHECK(
            cublasGemmEx(
                handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                token_num, token_num, HEAD_DIM,
                &rsqrt_head_dim,
                k, CUDA_R_16BF, n,
                q, CUDA_R_16BF, n,
                &zero,
                prob, CUDA_R_16BF, token_num,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT
            )
        );
    }

    launchAttentionSoftmaxPrefillKernel(Q_HEAD_NUM, token_num, attn_prob_buffer, stream);

    for (int q_head_idx = 0; q_head_idx < Q_HEAD_NUM; q_head_idx++) {
        const int v_head_idx = q_head_idx / (Q_HEAD_NUM / K_HEAD_NUM);
        const __nv_bfloat16* prob = attn_prob_buffer + q_head_idx * token_num * token_num;
        const __nv_bfloat16* v = v_begin + v_head_idx * HEAD_DIM;
        __nv_bfloat16* o = attn_output_buffer + q_head_idx * HEAD_DIM;
        CUBLAS_CHECK(
            cublasGemmEx(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                HEAD_DIM, token_num, token_num,
                &one,
                v, CUDA_R_16BF, n,
                prob, CUDA_R_16BF, token_num,
                &zero,
                o, CUDA_R_16BF, Q_PROJ_DIM,
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
            HIDDEN_DIM, token_num, Q_PROJ_DIM,
            &one,
            llama_weight.o_proj[layer_idx], CUDA_R_16BF, Q_PROJ_DIM,
            attn_output_buffer, CUDA_R_16BF, Q_PROJ_DIM,
            &zero,
            x_norm_buffer, CUDA_R_16BF, HIDDEN_DIM,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        )
    );

    launchResidualConnectionKernel(token_num, x_norm_buffer, hidden_state, stream);

    arena.reset();

    const int packed_dim = 2 * UP_PROJ_DIM;
    x_norm_buffer = arena.alloc<__nv_bfloat16>(token_num * HIDDEN_DIM);
    auto* gate_up_buffer = arena.alloc<__nv_bfloat16>(token_num * packed_dim);

    launchRMSNormKernel(token_num, eps, hidden_state, llama_weight.post_attention_layernorm[layer_idx], x_norm_buffer, stream);

    CUBLAS_CHECK(
        cublasGemmEx(handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            packed_dim, token_num, HIDDEN_DIM,
            &one,
            llama_weight.gate_up_proj[layer_idx], CUDA_R_16BF, HIDDEN_DIM,
            x_norm_buffer, CUDA_R_16BF, HIDDEN_DIM,
            &zero,
            gate_up_buffer, CUDA_R_16BF, packed_dim,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        )
    );

    launchSwiGLUInplaceKernel(token_num, UP_PROJ_DIM, packed_dim, gate_up_buffer, stream);

    CUBLAS_CHECK(
        cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            HIDDEN_DIM, token_num, UP_PROJ_DIM,
            &one,
            llama_weight.down_proj[layer_idx], CUDA_R_16BF, UP_PROJ_DIM,
            gate_up_buffer, CUDA_R_16BF, packed_dim,
            &zero,
            x_norm_buffer, CUDA_R_16BF, HIDDEN_DIM,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        )
    );

    launchResidualConnectionKernel(token_num, x_norm_buffer, hidden_state, stream);

    arena.reset();
}
