#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>


void launch_single_batch_store_kv_cache_kernel(
    std::size_t lines_to_store,
    int layer_idx,
    int line_offset,
    int kv_stride,
    const __nv_bfloat16* kv,
    __nv_bfloat16* kv_cache,
    cudaStream_t stream
);