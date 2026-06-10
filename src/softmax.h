#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>


void launch_attention_softmax_prefill_kernel(
    std::size_t head_count,
    std::size_t token_count,
    __nv_bfloat16* attention_scores,
    cudaStream_t stream
);
