#pragma once

#include <cuda_bf16.h>


void launchTokenEmbeddingKernel(
    const int* tokens,
    std::size_t token_num,
    const __nv_bfloat16* __restrict__ embedded_tokens,
    __nv_bfloat16* __restrict__ output
);

void launchRMSNormKernel(
    std::size_t token_num,
    float eps,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ output
);