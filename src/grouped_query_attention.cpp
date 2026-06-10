#include "grouped_query_attention.h"
#include "macro.h"




void grouped_query_attention(
    const Llama3_2& weights,
    int layer_idx,
    std::size_t token_count,
    __nv_bfloat16* hidden_state,
    cublasHandle_t cublas_handle
) {
    // Q/K/V Proj
    __nv_bfloat16* qkv_workspace = nullptr;
    // CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&qkv_workspace)));
    // projection(cublas_handle, token_count, Q_PROJ_DIM, HIDDEN_DIM, weights.q_proj[layer_idx], hidden_state, )
}
