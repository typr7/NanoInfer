#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>


void launch_rope_kernel(
    std::size_t token_count,
    int query_stride,
    int key_stride,
    __nv_bfloat16* query,
    __nv_bfloat16* key,
    cudaStream_t stream
);
