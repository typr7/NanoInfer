#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>


void launchRoPEKernel(
    std::size_t token_num,
    int ldq,
    int ldk,
    __nv_bfloat16* q,
    __nv_bfloat16* k,
    cudaStream_t stream
);
