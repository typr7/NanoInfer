#include <cuda_bf16.h>

#include "kernels.h"
#include "llama.h"


__global__
void embeddingGatherKernel(
    const int* tokens,
    __nv_bfloat16* embedding_buffer,
    __nv_bfloat16* embedded_tokens
) {
    const int dst = blockIdx.x * blockDim.x + threadIdx.x;
    const int src = tokens[blockIdx.x] * blockDim.x + threadIdx.x;
    embedding_buffer[dst] = embedded_tokens[src];
    embedding_buffer[dst + 1024] = embedded_tokens[src + 1024];
}

void launchEmbeddingGatherKernel(
    const int* tokens,
    std::size_t token_num,
    __nv_bfloat16* embedding_buffer,
    __nv_bfloat16* embedded_tokens
) {
    embeddingGatherKernel<<<token_num, 1024>>>(tokens, embedding_buffer, embedded_tokens);
}