#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>


void launch_token_embedding_kernel(
    std::size_t token_count,
    const std::int32_t* token_ids,
    const __nv_bfloat16* embedding_table,
    __nv_bfloat16* hidden_state,
    cudaStream_t stream
);

void launch_rms_norm_kernel(
    std::size_t token_count,
    float eps,
    const __nv_bfloat16* hidden_state,
    const __nv_bfloat16* norm_weights,
    __nv_bfloat16* normalized_state,
    cudaStream_t stream
);
