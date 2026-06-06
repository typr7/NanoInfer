#include <cublas_v2.h>

#include "llama.h"
#include "device_arena.hpp"
#include "kernels.h"


void projection(
    cublasHandle_t handle,
    int m, int n, int k,
    const __nv_bfloat16* proj_weight,
    const __nv_bfloat16* input,
    __nv_bfloat16* output
) {
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    cublasGemmEx(handle,
                 CUBLAS_OP_T,
                 CUBLAS_OP_N,
                 n, m, k,
                 &alpha,
                 proj_weight, CUDA_R_16BF, k,
                 input, CUDA_R_16BF, k,
                 &beta,
                 output, CUDA_R_16BF, n,
                 CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT);
}

void run_llama32_layer_prefill(
    cublasHandle_t handle,
    DeviceArena& arena,
    std::size_t token_num,
    int layer_idx,
    const Llama3_2& llama_weight,
    __nv_bfloat16* hidden_state
) {
    constexpr float eps   = 0.00001f;
    constexpr float alpha = 1.0f;
    constexpr float beta  = 0.0f;

    arena.reset();

    auto* x_norm_buffer = arena.alloc<__nv_bfloat16>(token_num * HIDDEN_DIM);
    auto* qkv_buffer = arena.alloc<__nv_bfloat16>(token_num * (Q_PROJ_DIM + 2 * K_PROJ_DIM));
    // auto* attn_score_buffer

    // rmsnorm
    launchRMSNormKernel(token_num, eps, hidden_state, llama_weight.input_layernorm[layer_idx], x_norm_buffer);

    // qkv proj
    int m = token_num;
    int n = Q_PROJ_DIM + 2 * K_PROJ_DIM;
    int k = HIDDEN_DIM;
    cublasGemmEx(handle,
                 CUBLAS_OP_T,
                 CUBLAS_OP_N,
                 n, m, k,
                 &alpha,
                 llama_weight.qkv_proj[layer_idx], CUDA_R_16BF, k,
                 x_norm_buffer, CUDA_R_16BF, k,
                 &beta,
                 qkv_buffer, CUDA_R_16BF, n,
                 CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT);
    
    // [token_num, q_proj_dim] --reshape--> [token_num, q_head_num, head_dim] --transpose--> [q_head_num, token_num, head_dim]
    // stride [q_proj_dim, 1]  --reshape--> [q_proj_dim, head_dim, 1]         --transpose--> [head_dim, q_proj_dim, 1]
    
    // RoPE
    launchRoPEKernel(token_num, Q_PROJ_DIM, qkv_buffer);
    launchRoPEKernel(token_num, K_PROJ_DIM, qkv_buffer + token_num * Q_PROJ_DIM * sizeof(__nv_bfloat16));

    for (int q_head_idx = 0; q_head_idx < Q_HEAD_NUM; q_head_idx++) {
        const int k_head_idx = q_head_idx / (Q_HEAD_NUM / K_HEAD_NUM);
        // const __nv_bfloat16* q = qkv_buffer
    }
}