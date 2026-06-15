#include "rms_norm.h"

#include "../llama.h"


namespace {

template <int BLOCK_SIZE> __global__
void rms_norm_kernel(
    const __nv_bfloat16* __restrict__ hidden_state,
    const __nv_bfloat16* __restrict__ norm_weights,
    __nv_bfloat16* __restrict__ normalized_state
) {
    __shared__ float rms_vec[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    float block_sum = 0.f;
    #pragma unroll
    for (int i = tid; i < HIDDEN_DIM; i += BLOCK_SIZE) {
        const int idx = bid * HIDDEN_DIM + i;
        const float val = static_cast<float>(hidden_state[idx]);
        block_sum += val * val;
    }
    rms_vec[tid] = block_sum;
    __syncthreads();

    #pragma unroll
    for (int offset = (BLOCK_SIZE >> 1); offset > 0; offset >>= 1) {
        if (tid < offset) {
            rms_vec[tid] += rms_vec[tid + offset];
        }
        __syncthreads();
    }
    const float rms_rfactor = rsqrtf(rms_vec[0] / HIDDEN_DIM + RMS_NORM_EPS);

    #pragma unroll
    for (int i = tid; i < HIDDEN_DIM; i += BLOCK_SIZE) {
        const int idx = bid * HIDDEN_DIM + i;
        const float val = static_cast<float>(hidden_state[idx]);
        const float norm_factor = static_cast<float>(norm_weights[i]);
        const float normed_val = norm_factor * rms_rfactor * val;
        normalized_state[idx] = static_cast<__nv_bfloat16>(normed_val);
    }
}

} // namespace

void launch_rms_norm_kernel(
    std::size_t token_count,
    const __nv_bfloat16* hidden_state,
    const __nv_bfloat16* norm_weights,
    __nv_bfloat16* normalized_state,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    rms_norm_kernel<BLOCK_SIZE>
        <<<token_count, BLOCK_SIZE, 0, stream>>>(hidden_state, norm_weights, normalized_state);
}