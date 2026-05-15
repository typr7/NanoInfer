#include <vector>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "macro.h"
#include "llama.h"
#include "kernels.h"


static constexpr std::size_t MAX_TOKEN_LEN = 512;

std::vector<int> inference(const std::vector<int>& input_tokens, const Llama3_2& weight)
{
    // embedding
    int* token_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&token_buffer), input_tokens.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(token_buffer, input_tokens.data(), input_tokens.size() * sizeof(int), cudaMemcpyHostToDevice));

    __nv_bfloat16* embedding_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&embedding_buffer), input_tokens.size() * 2048 * sizeof(__nv_bfloat16)));

    launchTokenEmbeddingKernel(token_buffer, input_tokens.size(), weight.embed_tokens, embedding_buffer);
}