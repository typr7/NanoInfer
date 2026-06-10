#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>


void launch_swiglu_inplace_kernel(
    std::size_t token_count,
    int activation_dim,
    int gate_up_stride,
    __nv_bfloat16* gate_up,
    cudaStream_t stream
);
