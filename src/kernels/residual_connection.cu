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
    const __nv_bfloat16* residual,
    __nv_bfloat16* hidden_state,
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
    const __nv_bfloat16* residual,
    __nv_bfloat16* hidden_state,
    cudaStream_t stream
) {
    residual_connection_kernel_v2<<<token_count, HALF_HIDDEN_DIM, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat162*>(residual),
        reinterpret_cast<__nv_bfloat162*>(hidden_state)
    );
}

// v3: single thread processes more data
template <int block_size>
__global__
void residual_connection_kernel_v3(
    const __nv_bfloat162* __restrict__ residual,
    __nv_bfloat162* __restrict__ hidden_state
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int bf162_per_thread = HALF_HIDDEN_DIM / block_size;
    const std::size_t idx = bid * HALF_HIDDEN_DIM + bf162_per_thread * tid;
    
    #pragma unroll
    for (int i = 0; i < bf162_per_thread; i++) {
        hidden_state[idx + i] = __hadd2(residual[idx + i], hidden_state[idx + i]);
    }
}

void launch_residual_connection_kernel_v3(
    std::size_t token_count,
    const __nv_bfloat16* residual,
    __nv_bfloat16* hidden_state,
    cudaStream_t stream
) {
    constexpr int block_size = 256;

    residual_connection_kernel_v3<block_size>
        <<<token_count, block_size, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat162*>(residual),
        reinterpret_cast<__nv_bfloat162*>(hidden_state)
    );
}

} // namespace

void launch_residual_connection_kernel(
    std::size_t token_count,
    const __nv_bfloat16* residual,
    __nv_bfloat16* hidden_state,
    cudaStream_t stream
) {
    launch_residual_connection_kernel_v3(token_count, residual, hidden_state, stream);
}
