#include "swiglu.h"
#include "llama.h"


__device__ __forceinline__
float SwiGLU(float gate, float up)
{
    return up * gate / (1.f + __expf(-gate));
}

__global__
void SwiGLUInplaceKernel(
    int dim,
    int ld,
    __nv_bfloat16* gate_up
) {
    __nv_bfloat16* gate = gate_up;
    __nv_bfloat16* up   = gate_up + dim;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    for (int i = tid; i < dim; i += blockDim.x) {
        const int idx = bid * ld + i;
        const float gate_val = static_cast<float>(gate[idx]);
        const float up_val = static_cast<float>(up[idx]);
        gate[idx] = static_cast<__nv_bfloat16>(SwiGLU(gate_val, up_val));
    }
}

// gate_up layout per token: [gate, up], gate = up * SiLU(gate)
void launchSwiGLUInplaceKernel(
    std::size_t token_num,
    int dim,
    int ld,
    __nv_bfloat16* gate_up,
    cudaStream_t stream
) {
    SwiGLUInplaceKernel<<<token_num, 256, 0, stream>>>(dim, ld, gate_up);
}
