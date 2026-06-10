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


std::vector<std::int32_t> inference(const std::vector<std::int32_t>& input_tokens, const Llama3_2& weights)
{
    const std::size_t token_count = input_tokens.size();
    if (token_count == 0) {
        return {};
    }

    cudaStream_t stream = nullptr;
    cublasHandle_t cublas_handle = nullptr;

    // embedding
    std::int32_t* token_ids_device = nullptr;
    __nv_bfloat16* hidden_state = nullptr;

    try {
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUBLAS_CHECK(cublasCreate(&cublas_handle));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&token_ids_device), token_count * sizeof(std::int32_t)));
        CUDA_CHECK(cudaMemcpyAsync(
            token_ids_device,
            input_tokens.data(),
            token_count * sizeof(std::int32_t),
            cudaMemcpyHostToDevice,
            stream
        ));

        // [token_count, hidden_dim]
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hidden_state), token_count * HIDDEN_DIM * sizeof(__nv_bfloat16)));
        launch_token_embedding_kernel(token_count, token_ids_device, weights.embed_tokens, hidden_state, stream);

        const std::size_t attention_workspace_count =
            token_count * HIDDEN_DIM +
            token_count * (Q_PROJ_DIM + 2 * K_PROJ_DIM) +
            Q_HEAD_NUM * token_count * token_count +
            token_count * Q_PROJ_DIM;
        const std::size_t mlp_workspace_count =
            token_count * HIDDEN_DIM +
            token_count * 2 * UP_PROJ_DIM;
        DeviceArena arena(std::max(attention_workspace_count, mlp_workspace_count) * sizeof(__nv_bfloat16));

        for (int layer_idx = 0; layer_idx < 16; layer_idx++) {
            run_llama_layer_prefill(
                weights,
                layer_idx,
                token_count,
                hidden_state,
                arena,
                stream,
                cublas_handle
            );
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
    } catch (...) {
        if (hidden_state) cudaFree(hidden_state);
        if (token_ids_device) cudaFree(token_ids_device);
        if (cublas_handle) cublasDestroy(cublas_handle);
        if (stream) cudaStreamDestroy(stream);
        throw;
    }

    CUDA_CHECK(cudaFree(hidden_state));
    CUDA_CHECK(cudaFree(token_ids_device));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUDA_CHECK(cudaStreamDestroy(stream));

    // TODO: implement decode stage
    return {};
}
