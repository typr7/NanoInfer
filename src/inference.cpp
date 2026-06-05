#include <vector>
#include <cstdint>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "macro.h"
#include "llama.h"
#include "kernels.h"


std::vector<std::int32_t> inference(const std::vector<std::int32_t>& input_tokens, const Llama3_2& weight)
{
    constexpr float eps = 0.00001f;

    std::size_t token_num = input_tokens.size();
    // embedding
    std::int32_t* token_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&token_buffer), token_num * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMemcpy(token_buffer, input_tokens.data(), token_num * sizeof(std::int32_t), cudaMemcpyHostToDevice));

    // [token_num, embedding_dim]
    __nv_bfloat16* embedding_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&embedding_buffer), token_num * HIDDEN_DIM * sizeof(__nv_bfloat16)));
    launchTokenEmbeddingKernel(token_num, token_buffer, weight.embed_tokens, embedding_buffer);

    for (int i = 0; i < 16; i++) {
        // RMSNorm 1
        launchRMSNormKernel(token_num, eps, embedding_buffer, weight.input_layernorm[i], embedding_buffer);


    }
}