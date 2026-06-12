#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>


void launch_single_batch_store_kv_cache_prefill_kernel(
    std::size_t kv_count,
    int kv_stride,
    int layer_idx,
    const __nv_bfloat16* kv,
    __nv_bfloat16* kv_cache,
    cudaStream_t stream
);