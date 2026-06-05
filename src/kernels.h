#pragma once

#include <cstdint>

#include <cuda_bf16.h>


void launchTokenEmbeddingKernel(
    std::size_t token_num,
    const std::int32_t* tokens,
    const __nv_bfloat16* embedded_tokens,
    __nv_bfloat16* output
);

void launchRMSNormKernel(
    std::size_t token_num,
    float eps,
    const __nv_bfloat16* input,
    const __nv_bfloat16* norm_weight,
    __nv_bfloat16* output
);

void launchRoPEKernel(std::size_t token_num, int proj_dim, __nv_bfloat16* input);