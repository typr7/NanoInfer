#pragma once


#include <array>
#include <utility>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "device_arena.hpp"

inline constexpr int MAX_TOKEN_LEN = 512;
inline constexpr int HIDDEN_DIM = 2048;
inline constexpr int HEAD_DIM = 64;

// GQA
inline constexpr int Q_PROJ_DIM = 2048;
inline constexpr int K_PROJ_DIM = 512;
inline constexpr int Q_HEAD_NUM = Q_PROJ_DIM / HEAD_DIM;
inline constexpr int K_HEAD_NUM = K_PROJ_DIM / HEAD_DIM;

// FFN
inline constexpr int UP_PROJ_DIM = 8192;
inline constexpr int DOWN_PROJ_DIM = 2048;

struct Llama3_2
{
    __nv_bfloat16* embed_tokens = nullptr;

    // 16 layers

        // self_attn
    std::array<__nv_bfloat16*, 16> qkv_proj = {};
    std::array<__nv_bfloat16*, 16> o_proj = {};

        // ffn
    std::array<__nv_bfloat16*, 16> gate_up_proj = {};
    std::array<__nv_bfloat16*, 16> down_proj = {};
    
        // RMSNorm
    std::array<__nv_bfloat16*, 16> input_layernorm          = {};
    std::array<__nv_bfloat16*, 16> post_attention_layernorm = {};

    __nv_bfloat16* norm = nullptr;

    __nv_bfloat16* lm_head = nullptr;

    // weight buffer
    __nv_bfloat16* weight    = nullptr;
    std::size_t weight_bytes = 0;

    Llama3_2() = default;
    
    Llama3_2(const Llama3_2&) = delete;

    Llama3_2& operator=(const Llama3_2&) = delete;

    Llama3_2(Llama3_2&& other) noexcept
    {
        *this = std::move(other);
    }

    Llama3_2& operator=(Llama3_2&& other) noexcept
    {
        if (this != &other) {
            if (weight) cudaFree(weight);

            embed_tokens = other.embed_tokens;
            qkv_proj = other.qkv_proj;
            o_proj = other.o_proj;

            gate_up_proj = other.gate_up_proj;
            down_proj = other.down_proj;

            input_layernorm = other.input_layernorm;
            post_attention_layernorm = other.post_attention_layernorm;

            norm = other.norm;
            lm_head = other.lm_head;
            weight = other.weight;
            weight_bytes = other.weight_bytes;

            other.weight = nullptr;
            other.weight_bytes = 0;
        }

        return *this;
    }

    ~Llama3_2() {
        if (weight) {
            cudaFree(weight);
            weight = nullptr;
            weight_bytes = 0;
        }
    }
};

void run_llama_layer_prefill(
    const Llama3_2& weights,
    int layer_idx,
    std::size_t token_count,
    __nv_bfloat16* hidden_state,
    DeviceArena& arena,
    cudaStream_t stream,
    cublasHandle_t cublas_handle
);
