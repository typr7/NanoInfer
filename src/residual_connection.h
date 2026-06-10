#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>


void launchResidualConnectionKernel(
    std::size_t token_num,
    const __nv_bfloat16* __restrict__ input1,
    __nv_bfloat16* __restrict__ input2_output,
    cudaStream_t stream
);
