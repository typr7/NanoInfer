#pragma once

#include <cuda_bf16.h>


void launchResidualConnectionKernelV1(
    std::size_t token_num,
    const __nv_bfloat16* __restrict__ input1,
    __nv_bfloat16* __restrict__ input2
);

void launchResidualConnectionKernelV2(
    std::size_t token_num,
    const __nv_bfloat16* __restrict__ input1,
    __nv_bfloat16* __restrict__ input2
);

void launchResidualConnectionKernelV3(
    std::size_t token_num,
    const __nv_bfloat16* __restrict__ input1,
    __nv_bfloat16* __restrict__ input2
);

inline
void launchResidualConnectionKernel(
    std::size_t token_num,
    const __nv_bfloat16* __restrict__ input1,
    __nv_bfloat16* __restrict__ input2
) {
    launchResidualConnectionKernelV1(token_num, input1, input2);
}
