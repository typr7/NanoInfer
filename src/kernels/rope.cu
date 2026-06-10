#include "rope.h"
#include "../llama.h"


namespace {

__global__
void rope_kernel(
    int query_stride,
    int key_stride,
    __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ key
) {
    const int query_idx = blockIdx.x * query_stride + 2 * threadIdx.x;
    const int rotary_dim = 2 * (threadIdx.x % (HEAD_DIM / 2));
    const float theta = blockIdx.x / powf(500000.f, static_cast<float>(rotary_dim) / HEAD_DIM);
    const float cos_theta = cosf(theta);
    const float sin_theta = sinf(theta);
    if (2 * threadIdx.x < Q_PROJ_DIM) {
        const float even_value = query[query_idx];
        const float odd_value = query[query_idx + 1];
        query[query_idx] = even_value * cos_theta - odd_value * sin_theta;
        query[query_idx + 1] = even_value * sin_theta + odd_value * cos_theta;
    }
    const int key_idx = blockIdx.x * key_stride + 2 * threadIdx.x;
    if (2 * threadIdx.x < K_PROJ_DIM) {
        const float even_value = key[key_idx];
        const float odd_value = key[key_idx + 1];
        key[key_idx] = even_value * cos_theta - odd_value * sin_theta;
        key[key_idx + 1] = even_value * sin_theta + odd_value * cos_theta;
    }
}

} // namespace

void launch_rope_kernel(
    std::size_t token_count,
    int query_stride,
    int key_stride,
    __nv_bfloat16* query,
    __nv_bfloat16* key,
    cudaStream_t stream
) {
    rope_kernel<<<token_count, HIDDEN_DIM / 2, 0, stream>>>(
        query_stride,
        key_stride,
        query,
        key
    );
}
