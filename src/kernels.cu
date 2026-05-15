#include <cmath>

#include <cuda_bf16.h>

#include "kernels.h"
#include "llama.h"


__global__
void tokenEmbeddingKernel(
    const int* tokens,
    const __nv_bfloat16* __restrict__ embedded_tokens,
    __nv_bfloat16* __restrict__ output
) {
    const int dst = blockIdx.x * 2048 + threadIdx.x;
    const int src = tokens[blockIdx.x] * 2048 + threadIdx.x;
    output[dst] = embedded_tokens[src];
    output[dst + 1024] = embedded_tokens[src + 1024];
}

void launchTokenEmbeddingKernel(
    const int* tokens,
    std::size_t token_num,
    const __nv_bfloat16* __restrict__ embedded_tokens,
    __nv_bfloat16* __restrict__ output
) {
    tokenEmbeddingKernel<<<token_num, 1024>>>(tokens, embedded_tokens, output);
}

__global__
void RMSNormKernel(
    float eps,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ output
) {
    __shared__ float rms_vec[1024];

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * 2048 + tid;
    const float val1 = static_cast<float>(input[idx]);
    const float val2 = static_cast<float>(input[idx + 1024]);
    rms_vec[tid] = val1 * val1 + val2 * val2;
    __syncthreads();

    #pragma unroll
    for (int offset = 512; offset > 0; offset >>= 1) {
        if (tid < offset) {
            rms_vec[tid] += rms_vec[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        rms_vec[0] = rsqrtf(rms_vec[0] / 2048.f + eps);
    }
    __syncthreads();

    output[idx] = static_cast<__nv_bfloat16>(val1 * rms_vec[0] * static_cast<float>(norm_weight[tid]));
    output[idx + 1024] = static_cast<__nv_bfloat16>(val2 * rms_vec[0] * static_cast<float>(norm_weight[tid + 1024]));
}

void launchRMSNormKernel(
    std::size_t token_num,
    float eps,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ output
) {
    RMSNormKernel<<<token_num, 1024>>>(eps, input, norm_weight, output);
}