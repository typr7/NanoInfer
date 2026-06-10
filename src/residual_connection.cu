#include "llama.h"
#include "residual_connection.h"


inline constexpr std::size_t HALF_HIDDEN_DIM = HIDDEN_DIM / 2;

// v1
__global__
void residualConnectionKernelV1(
    const __nv_bfloat16* __restrict__ input1,
    __nv_bfloat16* __restrict__ input2_output
) {
    const std::size_t idx = blockIdx.x * HIDDEN_DIM + threadIdx.x;
    input2_output[idx] += input1[idx];
    input2_output[idx + HALF_HIDDEN_DIM] += input1[idx + HALF_HIDDEN_DIM];
}

void launchResidualConnectionKernelV1(
    std::size_t token_num,
    const __nv_bfloat16* __restrict__ input1,
    __nv_bfloat16* __restrict__ input2_output,
    cudaStream_t stream
) {
    residualConnectionKernelV1<<<token_num, HALF_HIDDEN_DIM, 0, stream>>>(input1, input2_output);
}

// v2: combine two __nv_bfloat16 to __nv_bfloat162
__global__
void residualConnectionKernelV2(
    const __nv_bfloat162* __restrict__ input1,
    __nv_bfloat162* __restrict__ input2_output
) {
    const std::size_t idx = blockIdx.x * HALF_HIDDEN_DIM + threadIdx.x;
    input2_output[idx] = __hadd2(input1[idx], input2_output[idx]);
}

void launchResidualConnectionKernelV2(
    std::size_t token_num,
    const __nv_bfloat16* __restrict__ input1,
    __nv_bfloat16* __restrict__ input2_output,
    cudaStream_t stream
) {
    residualConnectionKernelV2<<<token_num, HALF_HIDDEN_DIM, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat162*>(input1),
        reinterpret_cast<__nv_bfloat162*>(input2_output)
    );
}

// v3: single thread processes more data
__global__
void residualConnectionKernelV3(
    const __nv_bfloat162* __restrict__ input1,
    __nv_bfloat162* __restrict__ input2_output
) {
    const int bf162_per_thread = HALF_HIDDEN_DIM / blockDim.x;
    const std::size_t idx = blockIdx.x * HALF_HIDDEN_DIM + bf162_per_thread * threadIdx.x;
    
    #pragma unroll
    for (int i = 0; i < bf162_per_thread; i++) {
        input2_output[idx + i] = __hadd2(input1[idx + i], input2_output[idx + i]);
    }
}

void launchResidualConnectionKernelV3(
    std::size_t token_num,
    const __nv_bfloat16* __restrict__ input1,
    __nv_bfloat16* __restrict__ input2_output,
    cudaStream_t stream
) {
    residualConnectionKernelV3<<<token_num, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat162*>(input1),
        reinterpret_cast<__nv_bfloat162*>(input2_output)
    );
}

void launchResidualConnectionKernel(
    std::size_t token_num,
    const __nv_bfloat16* __restrict__ input1,
    __nv_bfloat16* __restrict__ input2_output,
    cudaStream_t stream
) {
    launchResidualConnectionKernelV3(token_num, input1, input2_output, stream);
}
