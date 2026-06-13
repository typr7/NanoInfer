#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>


void launch_rms_norm_kernel(
    std::size_t token_count,
    const __nv_bfloat16* hidden_state,
    const __nv_bfloat16* norm_weights,
    __nv_bfloat16* normalized_state,
    cudaStream_t stream
);