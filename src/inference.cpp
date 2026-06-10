#include <vector>
#include <cstdint>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include "macro.h"
#include "llama.h"
#include "kernels.h"
#include "device_arena.hpp"


std::vector<std::int32_t> inference(const std::vector<std::int32_t>& input_tokens, const Llama3_2& weight)
{
    std::size_t token_num = input_tokens.size();
    if (token_num == 0) {
        return {};
    }

    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;

    // embedding
    std::int32_t* token_buffer = nullptr;
    __nv_bfloat16* embedding_buffer = nullptr;

    try {
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUBLAS_CHECK(cublasCreate(&handle));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&token_buffer), token_num * sizeof(std::int32_t)));
        CUDA_CHECK(cudaMemcpyAsync(
            token_buffer,
            input_tokens.data(),
            token_num * sizeof(std::int32_t),
            cudaMemcpyHostToDevice,
            stream
        ));

        // [token_num, embedding_dim]
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&embedding_buffer), token_num * HIDDEN_DIM * sizeof(__nv_bfloat16)));
        launchTokenEmbeddingKernel(token_num, token_buffer, weight.embed_tokens, embedding_buffer, stream);

        const std::size_t attn_workspace_count =
            token_num * HIDDEN_DIM +
            token_num * (Q_PROJ_DIM + 2 * K_PROJ_DIM) +
            Q_HEAD_NUM * token_num * token_num +
            token_num * Q_PROJ_DIM;
        const std::size_t mlp_workspace_count =
            token_num * HIDDEN_DIM +
            token_num * 2 * UP_PROJ_DIM;
        DeviceArena arena(std::max(attn_workspace_count, mlp_workspace_count) * sizeof(__nv_bfloat16));

        for (int i = 0; i < 16; i++) {
            run_llama32_layer_prefill(arena, token_num, i, weight, embedding_buffer, stream, handle);
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
    } catch (...) {
        if (embedding_buffer) cudaFree(embedding_buffer);
        if (token_buffer) cudaFree(token_buffer);
        if (handle) cublasDestroy(handle);
        if (stream) cudaStreamDestroy(stream);
        throw;
    }

    CUDA_CHECK(cudaFree(embedding_buffer));
    CUDA_CHECK(cudaFree(token_buffer));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));

    // TODO: implement decode stage
    return {};
}
