#include "swiglu.h"
#include "../llama.h"


namespace {

__device__ __forceinline__
float swiglu(float gate, float up)
{
    return up * gate / (1.f + expf(-gate));
}

template <int BLOCK_SIZE>
__global__
void swiglu_inplace_kernel(
    int activation_dim,
    int gate_up_stride,
    __nv_bfloat16* gate_up
) {
    __nv_bfloat16* gate = gate_up;
    __nv_bfloat16* up   = gate_up + activation_dim;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    for (int i = tid; i < activation_dim; i += BLOCK_SIZE) {
        const int idx = bid * gate_up_stride + i;
        const float gate_val = static_cast<float>(gate[idx]);
        const float up_val = static_cast<float>(up[idx]);
        gate[idx] = static_cast<__nv_bfloat16>(swiglu(gate_val, up_val));
    }
}

} // namespace

// gate_up layout per token: [gate, up], gate = up * SiLU(gate)
void launch_swiglu_inplace_kernel(
    std::size_t token_count,
    int activation_dim,
    int gate_up_stride,
    __nv_bfloat16* gate_up,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;

    swiglu_inplace_kernel<BLOCK_SIZE>
        <<<token_count, BLOCK_SIZE, 0, stream>>>(
        activation_dim,
        gate_up_stride,
        gate_up
    );
}
