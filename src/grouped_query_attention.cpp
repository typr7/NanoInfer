#include <vector>

#include <cublas_v2.h>

#include "grouped_query_attention.h"
#include "llama.h"
#include "macro.h"




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