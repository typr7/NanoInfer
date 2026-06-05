#include <vector>

#include <cublas_v2.h>

#include "grouped_query_attention.h"
#include "llama.h"
#include "macro.h"


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

void groupedQueryAttention(
    std::size_t token_num,
    int layer_idx,
    const Llama3_2& weight,
    cublasHandle_t handle,
    __nv_bfloat16* hidden_state
) {
    // Q/K/V Proj
    __nv_bfloat16* tmp_buffer = nullptr;
    // CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&tmp_buffer)));
    // projection(handle, token_num, Q_PROJ_DIM, HIDDEN_DIM, weight.q_proj[layer_idx], hidden_state, )
}