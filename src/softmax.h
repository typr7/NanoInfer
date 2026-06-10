#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>


void launchAttentionSoftmaxPrefillKernel(
    std::size_t attn_head_num,
    std::size_t token_num,
    __nv_bfloat16* input_output,
    cudaStream_t stream
);
