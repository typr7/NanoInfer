#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>


void launchSwiGLUInplaceKernel(
    std::size_t token_num,
    int dim,
    int ld,
    __nv_bfloat16* gate_up,
    cudaStream_t stream
);
