#include "kv_cache.h"
#include "../llama.h"
#include "../helper.hpp"


namespace
{

template <int block_size> __global__
void single_batch_store_kv_cache_kernel(
    int layer_idx,
    int line_offset,
    int kv_stride,
    const __nv_bfloat16* __restrict__ kv,
    __nv_bfloat16* __restrict__ kv_cache
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    constexpr int cache_stride = 2 * K_PROJ_DIM;

    const __nv_bfloat16* src = kv + bid * kv_stride;
    __nv_bfloat16* dst  = kv_cache_layer_begin(kv_cache, layer_idx)
                        + (line_offset + blockIdx.x) * cache_stride;

    #pragma unroll
    for (int i = tid; i < cache_stride; i += block_size) {
        dst[i] = src[i];
    }
}

}

void launch_single_batch_store_kv_cache_kernel(
    std::size_t lines_to_store,
    int layer_idx,
    int line_offset,
    int kv_stride,
    const __nv_bfloat16* kv,
    __nv_bfloat16* kv_cache,
    cudaStream_t stream
) {
    constexpr int block_size = 256;
    single_batch_store_kv_cache_kernel<block_size>
        <<<lines_to_store, block_size, 0, stream>>>(layer_idx, line_offset, kv_stride, kv, kv_cache);
}
