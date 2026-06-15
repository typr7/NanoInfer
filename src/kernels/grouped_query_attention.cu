#include <cmath>

#include <math_constants.h>

#include "grouped_query_attention.h"
#include "../llama.h"
#include "../helper.hpp"


namespace
{

template <int block_size> __global__
void single_batch_gqa_decode_kernel_v1(
    std::size_t token_count,
    int layer_idx,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ kv_cache,
    __nv_bfloat16* __restrict__ attn_output
) {
    __shared__ float prob_vec[MAX_TOKEN_LEN];
    __shared__ float reduce_vec[block_size];

    constexpr int group_size = Q_HEAD_NUM / K_HEAD_NUM;
    constexpr int cache_stride = 2 * K_PROJ_DIM;

    const float rsqrt_head_dim = rsqrtf(static_cast<float>(HEAD_DIM));
    const int k_head_idx = blockIdx.x;
    const int q_head_idx = k_head_idx * group_size + blockIdx.y;
    const int tid = threadIdx.x;

    const __nv_bfloat16* query_head  = q + q_head_idx * HEAD_DIM;
    const __nv_bfloat16* key_head    = kv_cache_layer_begin(kv_cache, layer_idx)
                                     + k_head_idx * HEAD_DIM;
    const __nv_bfloat16* value_head  = key_head + K_PROJ_DIM;
    __nv_bfloat16* output_head = attn_output + q_head_idx * HEAD_DIM;
    
    // QK^T / sqrt(head_dim)
    for (int i = tid; i < token_count; i += block_size) {
        float val = 0.f;
        for (int j = 0; j < HEAD_DIM; j++) {
            val += static_cast<float>(query_head[j]) 
                 * static_cast<float>(key_head[i * cache_stride + j]);
        }
        prob_vec[i] = val * rsqrt_head_dim;
    }
    __syncthreads();

    // max reduction
    float local_max = -CUDART_INF_F;
    for (int i = tid; i < token_count; i += block_size) {
        local_max = fmaxf(local_max, prob_vec[i]);
    }
    reduce_vec[tid] = local_max;
    __syncthreads();

    #pragma unroll
    for (int offset = block_size >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            reduce_vec[tid] = fmaxf(reduce_vec[tid], reduce_vec[tid + offset]);
        }
        __syncthreads();
    }
    const float global_max = reduce_vec[0];

    for (int i = tid; i < token_count; i += block_size) {
        prob_vec[i] = expf(prob_vec[i] - global_max);
    }
    __syncthreads();

    // sum softmax
    float local_sum = 0.f;
    for (int i = tid; i < token_count; i += block_size) {
        local_sum += prob_vec[i];
    }
    reduce_vec[tid] = local_sum;
    __syncthreads();

    #pragma unroll
    for (int offset = block_size >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            reduce_vec[tid] += reduce_vec[tid + offset];
        }
        __syncthreads();
    }
    const float global_sum = reduce_vec[0];

    for (int i = tid; i < token_count; i += block_size) {
        prob_vec[i] /= global_sum;
    }
    __syncthreads();

    constexpr int lanes_per_dim = block_size / HEAD_DIM;
    const int dim = tid % HEAD_DIM;
    const int lane = tid / HEAD_DIM;
    
    float acc = 0.f;
    for (int j = lane; j < token_count; j += lanes_per_dim) {
        acc += prob_vec[j] * static_cast<float>(value_head[j * cache_stride + dim]);
    }
    reduce_vec[tid] = acc;
    __syncthreads();

    if (lane == 0) {
        acc = 0.f;
        #pragma unroll
        for (int i = dim; i < block_size; i += HEAD_DIM) {
            acc += reduce_vec[i];
        }
        output_head[dim] = static_cast<__nv_bfloat16>(acc);
    }
}

}

void launch_single_batch_gqa_decode_kernel(
    std::size_t token_count,
    int layer_idx,
    const __nv_bfloat16* q,
    const __nv_bfloat16* kv_cache,
    __nv_bfloat16* attn_output,
    cudaStream_t stream
) {
    constexpr int group_size = Q_HEAD_NUM / K_HEAD_NUM;
    constexpr int block_size = 256;

    single_batch_gqa_decode_kernel_v1<block_size>
        <<<dim3(K_HEAD_NUM, group_size), block_size, 0, stream>>>(
        token_count,
        layer_idx,
        q,
        kv_cache,
        attn_output
    );
}