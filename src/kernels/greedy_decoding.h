#pragma once

#include <cstdint>

#include <cuda_runtime.h>
#include <cuda_bf16.h>


void launch_single_batch_greedy_decode_kernel(
    const __nv_bfloat16* logits,
    const __nv_bfloat16* embedding_table,
    __nv_bfloat16* next_token_embedding,
    int32_t* next_token_id,
    cudaStream_t stream
);
