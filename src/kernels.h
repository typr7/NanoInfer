#pragma once

#include <cuda_bf16.h>


void launchEmbeddingGatherKernel(const int* tokens, std::size_t token_num, __nv_bfloat16* embedding_buffer, __nv_bfloat16* embedded_tokens);