#pragma once

#include <cstddef>

#include <cuda_bf16.h>
#include <cublas_v2.h>

#include "llama.h"


void grouped_query_attention(
    const Llama3_2& weights,
    int layer_idx,
    std::size_t token_count,
    __nv_bfloat16* hidden_state,
    cublasHandle_t cublas_handle
);
