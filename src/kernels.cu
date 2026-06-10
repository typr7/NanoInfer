#include <cmath>
#include <stdexcept>

#include <cuda_bf16.h>
#include <cublas_v2.h>

#include "kernels.h"
#include "llama.h"


__global__
void token_embedding_kernel(
    const std::int32_t* __restrict__ token_ids,
    const __nv_bfloat16* __restrict__ embedding_table,
    __nv_bfloat16* __restrict__ hidden_state
) {
    const int dst = blockIdx.x * HIDDEN_DIM + threadIdx.x;
    const int src = token_ids[blockIdx.x] * HIDDEN_DIM + threadIdx.x;
    hidden_state[dst] = embedding_table[src];
    hidden_state[dst + HIDDEN_DIM / 2] = embedding_table[src + HIDDEN_DIM / 2];
}

void launch_token_embedding_kernel(
    std::size_t token_count,
    const std::int32_t* token_ids,
    const __nv_bfloat16* embedding_table,
    __nv_bfloat16* hidden_state,
    cudaStream_t stream
) {
    token_embedding_kernel<<<token_count, HIDDEN_DIM / 2, 0, stream>>>(
        token_ids,
        embedding_table,
        hidden_state
    );
}

__global__
void rms_norm_kernel(
    float eps,
    const __nv_bfloat16* hidden_state,
    const __nv_bfloat16* __restrict__ norm_weights,
    __nv_bfloat16* normalized_state
) {
    __shared__ float rms_vec[HIDDEN_DIM / 2];

    const int thread_idx = threadIdx.x;
    const int idx = blockIdx.x * HIDDEN_DIM + thread_idx;
    const float lo_value = static_cast<float>(hidden_state[idx]);
    const float hi_value = static_cast<float>(hidden_state[idx + HIDDEN_DIM / 2]);
    rms_vec[thread_idx] = lo_value * lo_value + hi_value * hi_value;
    __syncthreads();

    #pragma unroll
    for (int offset = HIDDEN_DIM / 4; offset > 0; offset >>= 1) {
        if (thread_idx < offset) {
            rms_vec[thread_idx] += rms_vec[thread_idx + offset];
        }
        __syncthreads();
    }

    if (thread_idx == 0) {
        rms_vec[0] = rsqrtf(rms_vec[0] / HIDDEN_DIM + eps);
    }
    __syncthreads();

    normalized_state[idx] = static_cast<__nv_bfloat16>(
        lo_value * rms_vec[0] * static_cast<float>(norm_weights[thread_idx])
    );
    normalized_state[idx + HIDDEN_DIM / 2] = static_cast<__nv_bfloat16>(
        hi_value * rms_vec[0] * static_cast<float>(norm_weights[thread_idx + HIDDEN_DIM / 2])
    );
}

void launch_rms_norm_kernel(
    std::size_t token_count,
    float eps,
    const __nv_bfloat16* hidden_state,
    const __nv_bfloat16* norm_weights,
    __nv_bfloat16* normalized_state,
    cudaStream_t stream
) {
    rms_norm_kernel<<<token_count, HIDDEN_DIM / 2, 0, stream>>>(
        eps,
        hidden_state,
        norm_weights,
        normalized_state
    );
}
