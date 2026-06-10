#include <cuda_bf16.h>

#include "softmax.h"


__global__
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

    #pragma unroll
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

    #pragma unroll
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

void launch_attention_softmax_prefill_kernel(
    std::size_t head_count,
    std::size_t token_count,
    __nv_bfloat16* attention_scores,
    cudaStream_t stream
) {
    attention_softmax_prefill_kernel_v1<<<head_count * token_count, token_count, 0, stream>>>(
        attention_scores
    );
}
