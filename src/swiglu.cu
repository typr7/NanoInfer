#include "swiglu.h"
#include "llama.h"


__device__ __forceinline__
float swiglu(float gate, float up)
{
    return up * gate / (1.f + __expf(-gate));
}

__global__
void swiglu_inplace_kernel(
    int activation_dim,
    int gate_up_stride,
    __nv_bfloat16* gate_up
) {
    __nv_bfloat16* gate = gate_up;
    __nv_bfloat16* up   = gate_up + activation_dim;

    const int thread_idx = threadIdx.x;
    const int token_idx = blockIdx.x;

    for (int channel_idx = thread_idx; channel_idx < activation_dim; channel_idx += blockDim.x) {
        const int idx = token_idx * gate_up_stride + channel_idx;
        const float gate_val = static_cast<float>(gate[idx]);
        const float up_val = static_cast<float>(up[idx]);
        gate[idx] = static_cast<__nv_bfloat16>(swiglu(gate_val, up_val));
    }
}

// gate_up layout per token: [gate, up], gate = up * SiLU(gate)
void launch_swiglu_inplace_kernel(
    std::size_t token_count,
    int activation_dim,
    int gate_up_stride,
    __nv_bfloat16* gate_up,
    cudaStream_t stream
) {
    swiglu_inplace_kernel<<<token_count, 256, 0, stream>>>(
        activation_dim,
        gate_up_stride,
        gate_up
    );
}
