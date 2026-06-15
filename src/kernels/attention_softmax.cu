#include <math_constants.h>
#include <cuda_bf16.h>

#include "attention_softmax.h"
#include "../llama.h"


namespace {

[[maybe_unused]] __global__
void attention_softmax_prefill_kernel_v1(__nv_bfloat16* attention_scores)
{
    __shared__ float vec[1024];

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int row_dim = blockIdx.x % blockDim.x + 1;

    if ((threadIdx.x & 0b01) == 0 && threadIdx.x + 1 < row_dim) {
        vec[threadIdx.x] = fmaxf(attention_scores[idx], attention_scores[idx + 1]);
    }
    if (threadIdx.x == 0 && (row_dim & 0b01) == 1) {
        vec[row_dim - 1] = attention_scores[blockIdx.x * blockDim.x + row_dim - 1];
    }
    __syncthreads();

    for (int i = 2; i < row_dim; i <<= 1) {
        if ((threadIdx.x & ((i << 1) - 1)) == 0 && threadIdx.x + i < row_dim) {
            vec[threadIdx.x] = fmaxf(vec[threadIdx.x], vec[threadIdx.x + i]);
        }
        __syncthreads();
    }   

    const float row_max = vec[0];
    float exp_value = 0.f;
    if (threadIdx.x < row_dim) {
        exp_value = expf(static_cast<float>(attention_scores[idx]) - row_max);
        vec[threadIdx.x] = exp_value;
    }
    __syncthreads();

    for(int active = row_dim; active > 1; active = (active + 1) >> 1) {
        const int half = active >> 1;
        const int offset = (active + 1) >> 1;
        if (threadIdx.x < half) {
            vec[threadIdx.x] += vec[threadIdx.x + offset];
        }
        __syncthreads();
    }

    __nv_bfloat16 softmax_value = 0.f;
    if (threadIdx.x < row_dim) {
        softmax_value = static_cast<__nv_bfloat16>(exp_value / vec[0]);
    }
    attention_scores[idx] = softmax_value;
}

template <int block_size> __global__
void attention_softmax_prefill_kernel_v2(std::size_t token_count, __nv_bfloat16* attn_scores)
{
    __shared__ float prob_vec[MAX_TOKEN_LEN];
    __shared__ float reduce_vec[block_size];

    const int tid = threadIdx.x;
    const int head_idx = blockIdx.x;
    const int row_idx = blockIdx.y;

    __nv_bfloat16* vec_begin = attn_scores
                             + head_idx * token_count * token_count
                             + row_idx * token_count;

    // max reduction
    float local_max = -CUDART_INF_F;
    for (int i = tid; i < row_idx + 1; i += block_size) {
        local_max = fmaxf(local_max, static_cast<float>(vec_begin[i]));
    }
    reduce_vec[tid] = local_max;
    __syncthreads();

    #pragma unroll
    for (int offset = (block_size >> 1); offset > 0; offset >>= 1) {
        if (tid < offset) {
            reduce_vec[tid] = fmaxf(reduce_vec[tid], reduce_vec[tid + offset]);
        }
        __syncthreads();
    }
    const float global_max = reduce_vec[0];

    for (int i = tid; i < row_idx + 1; i += block_size) {
        prob_vec[i] = expf(static_cast<float>(vec_begin[i]) - global_max);
    }

    // sum reduction
    float local_sum = 0.f;
    for (int i = tid; i < row_idx + 1; i += block_size) {
        local_sum += prob_vec[i];
    }
    reduce_vec[tid] = local_sum;
    __syncthreads();

    #pragma unroll
    for (int offset = (block_size >> 1); offset > 0; offset >>= 1) {
        if (tid < offset) {
            reduce_vec[tid] += reduce_vec[tid + offset];
        }
        __syncthreads();
    }
    const float global_sum = reduce_vec[0];

    for (int i = tid; i < token_count; i += block_size) {
        vec_begin[i] = __nv_bfloat16(0.f);
        if (i < row_idx + 1) {
            vec_begin[i] = static_cast<__nv_bfloat16>(prob_vec[i] / global_sum);
        }
    }
}

} // namespace

void launch_attention_softmax_prefill_kernel(
    std::size_t head_count,
    std::size_t token_count,
    __nv_bfloat16* attention_scores,
    cudaStream_t stream
) {
    constexpr int block_size = 256;

    attention_softmax_prefill_kernel_v2<block_size>
        <<<dim3(head_count, token_count), block_size, 0, stream>>>(token_count, attention_scores);
}
