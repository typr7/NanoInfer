#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>


void launchTokenEmbeddingKernel(
    std::size_t token_num,
    const std::int32_t* tokens,
    const __nv_bfloat16* embedded_tokens,
    __nv_bfloat16* output,
    cudaStream_t stream
);

void launchRMSNormKernel(
    std::size_t token_num,
    float eps,
    const __nv_bfloat16* input,
    const __nv_bfloat16* norm_weight,
    __nv_bfloat16* output,
    cudaStream_t stream
);
