#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>


void launch_single_batch_gqa_decode_kernel(
    std::size_t token_count,
    int layer_idx,
    const __nv_bfloat16* q,
    const __nv_bfloat16* kv_cache,
    __nv_bfloat16* attn_output,
    cudaStream_t stream
);