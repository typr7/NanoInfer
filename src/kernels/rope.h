#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>


void launch_rope_llama3_half_split_inplace_kernel(
    std::size_t token_count,
    int position_offset,
    int qk_stride,
    __nv_bfloat16* qk,
    cudaStream_t stream
);
