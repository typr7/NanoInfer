#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>


void launch_residual_connection_kernel(
    std::size_t token_count,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ hidden_state,
    cudaStream_t stream
);
