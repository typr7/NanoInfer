#include <cuda_bf16.h>

#include "softmax.h"


__global__
void attentionSoftmaxPrefillKernelV1(__nv_bfloat16* input_output)
{
    __shared__ float vec[1024];

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int row_dim = blockIdx.x % blockDim.x + 1;

    if ((threadIdx.x & 0b01) == 0 && threadIdx.x + 1 < row_dim) {
        vec[threadIdx.x] = fmaxf(input_output[idx], input_output[idx + 1]);
    }
    if (threadIdx.x == 0 && (row_dim & 0b01) == 1) {
        vec[row_dim - 1] = input_output[blockIdx.x * blockDim.x + row_dim - 1];
    }
    __syncthreads();

    #pragma unroll
    for (int i = 2; i < row_dim; i <<= 1) {
        if ((threadIdx.x & ((i << 1) - 1)) == 0 && threadIdx.x + i < row_dim) {
            vec[threadIdx.x] = fmaxf(vec[threadIdx.x], vec[threadIdx.x + i]);
        }
        __syncthreads();
    }   

    const float maxx = vec[0];
    float ex = 0.f;
    if (threadIdx.x < row_dim) {
        vec[threadIdx.x] = ex = expf(static_cast<float>(input_output[idx]) - maxx);
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

    __nv_bfloat16 res = 0.f;
    if (threadIdx.x < row_dim) {
        res = static_cast<__nv_bfloat16>(ex / vec[0]);
    }
    input_output[idx] = res;
}

void launchAttentionSoftmaxPrefillKernel(
    std::size_t attn_head_num,
    std::size_t token_num,
    __nv_bfloat16* input_output,
    cudaStream_t stream
) {
    attentionSoftmaxPrefillKernelV1<<<attn_head_num * token_num, token_num, 0, stream>>>(input_output);
}
