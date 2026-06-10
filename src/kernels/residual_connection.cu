#include "../llama.h"
#include "residual_connection.h"


namespace {

inline constexpr std::size_t HALF_HIDDEN_DIM = HIDDEN_DIM / 2;

// v1
__global__
void residual_connection_kernel_v1(
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ hidden_state
) {
    const std::size_t idx = blockIdx.x * HIDDEN_DIM + threadIdx.x;
    hidden_state[idx] += residual[idx];
    hidden_state[idx + HALF_HIDDEN_DIM] += residual[idx + HALF_HIDDEN_DIM];
}

[[maybe_unused]] void launch_residual_connection_kernel_v1(
    std::size_t token_count,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ hidden_state,
    cudaStream_t stream
) {
    residual_connection_kernel_v1<<<token_count, HALF_HIDDEN_DIM, 0, stream>>>(residual, hidden_state);
}

// v2: combine two __nv_bfloat16 to __nv_bfloat162
__global__
void residual_connection_kernel_v2(
    const __nv_bfloat162* __restrict__ residual,
    __nv_bfloat162* __restrict__ hidden_state
) {
    const std::size_t idx = blockIdx.x * HALF_HIDDEN_DIM + threadIdx.x;
    hidden_state[idx] = __hadd2(residual[idx], hidden_state[idx]);
}

[[maybe_unused]] void launch_residual_connection_kernel_v2(
    std::size_t token_count,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ hidden_state,
    cudaStream_t stream
) {
    residual_connection_kernel_v2<<<token_count, HALF_HIDDEN_DIM, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat162*>(residual),
        reinterpret_cast<__nv_bfloat162*>(hidden_state)
    );
}

// v3: single thread processes more data
__global__
void residual_connection_kernel_v3(
    const __nv_bfloat162* __restrict__ residual,
    __nv_bfloat162* __restrict__ hidden_state
) {
    const int bf162_per_thread = HALF_HIDDEN_DIM / blockDim.x;
    const std::size_t idx = blockIdx.x * HALF_HIDDEN_DIM + bf162_per_thread * threadIdx.x;
    
    #pragma unroll
    for (int i = 0; i < bf162_per_thread; i++) {
        hidden_state[idx + i] = __hadd2(residual[idx + i], hidden_state[idx + i]);
    }
}

void launch_residual_connection_kernel_v3(
    std::size_t token_count,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ hidden_state,
    cudaStream_t stream
) {
    residual_connection_kernel_v3<<<token_count, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat162*>(residual),
        reinterpret_cast<__nv_bfloat162*>(hidden_state)
    );
}

} // namespace

void launch_residual_connection_kernel(
    std::size_t token_count,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ hidden_state,
    cudaStream_t stream
) {
    launch_residual_connection_kernel_v3(token_count, residual, hidden_state, stream);
}
