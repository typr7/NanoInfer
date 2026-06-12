#include "kv_cache.h"
#include "../llama.h"


namespace
{

template <int BLOCK_SIZE> __global__
void single_batch_store_kv_cache_prefill_kernel(
    int kv_stride,
    int layer_idx,
    const __nv_bfloat16* __restrict__ kv,
    __nv_bfloat16* __restrict__ kv_cache
) {
    constexpr int line_dim = 2 * K_PROJ_DIM;
    constexpr int layer_stride = MAX_TOKEN_LEN * line_dim;

    const __nv_bfloat16* src = kv + blockIdx.x * kv_stride;
    __nv_bfloat16* dst = kv_cache + layer_idx * layer_stride + blockIdx.x * line_dim;

    const int tid = threadIdx.x;

    #pragma unroll
    for (int i = tid; i < line_dim; i += BLOCK_SIZE) {
        dst[i] = src[i];
    }
}

}

void launch_single_batch_store_kv_cache_prefill_kernel(
    std::size_t kv_count,
    int kv_stride,
    int layer_idx,
    const __nv_bfloat16* kv,
    __nv_bfloat16* kv_cache,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    single_batch_store_kv_cache_prefill_kernel<BLOCK_SIZE>
        <<<kv_count, BLOCK_SIZE, 0, stream>>>(kv_stride, layer_idx, kv, kv_cache);
}