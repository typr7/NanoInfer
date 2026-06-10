#include <cmath>
#include <stdexcept>

#include <cuda_bf16.h>
#include <cublas_v2.h>

#include "kernels.h"
#include "llama.h"


__global__
void tokenEmbeddingKernel(
    const std::int32_t* __restrict__ tokens,
    const __nv_bfloat16* __restrict__ embedded_tokens,
    __nv_bfloat16* __restrict__ output
) {
    const int dst = blockIdx.x * HIDDEN_DIM + threadIdx.x;
    const int src = tokens[blockIdx.x] * HIDDEN_DIM + threadIdx.x;
    output[dst] = embedded_tokens[src];
    output[dst + HIDDEN_DIM / 2] = embedded_tokens[src + HIDDEN_DIM / 2];
}

void launchTokenEmbeddingKernel(
    std::size_t token_num,
    const std::int32_t* tokens,
    const __nv_bfloat16* embedded_tokens,
    __nv_bfloat16* output,
    cudaStream_t stream
) {
    tokenEmbeddingKernel<<<token_num, HIDDEN_DIM / 2, 0, stream>>>(tokens, embedded_tokens, output);
}

__global__
void RMSNormKernel(
    float eps,
    const __nv_bfloat16* input,
    const __nv_bfloat16* __restrict__ norm_weight,
    __nv_bfloat16* output
) {
    __shared__ float rms_vec[HIDDEN_DIM / 2];

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * HIDDEN_DIM + tid;
    const float val1 = static_cast<float>(input[idx]);
    const float val2 = static_cast<float>(input[idx + HIDDEN_DIM / 2]);
    rms_vec[tid] = val1 * val1 + val2 * val2;
    __syncthreads();

    #pragma unroll
    for (int offset = HIDDEN_DIM / 4; offset > 0; offset >>= 1) {
        if (tid < offset) {
            rms_vec[tid] += rms_vec[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        rms_vec[0] = rsqrtf(rms_vec[0] / HIDDEN_DIM + eps);
    }
    __syncthreads();

    output[idx] = static_cast<__nv_bfloat16>(val1 * rms_vec[0] * static_cast<float>(norm_weight[tid]));
    output[idx + HIDDEN_DIM / 2] = static_cast<__nv_bfloat16>(val2 * rms_vec[0] * static_cast<float>(norm_weight[tid + HIDDEN_DIM / 2]));
}

void launchRMSNormKernel(
    std::size_t token_num,
    float eps,
    const __nv_bfloat16* input,
    const __nv_bfloat16* norm_weight,
    __nv_bfloat16* output,
    cudaStream_t stream
) {
    RMSNormKernel<<<token_num, HIDDEN_DIM / 2, 0, stream>>>(eps, input, norm_weight, output);
}
