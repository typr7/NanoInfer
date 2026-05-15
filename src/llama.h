#pragma once


#include <array>

#include <cuda_runtime.h>
#include <cuda_bf16.h>


struct Llama3_2
{
    __nv_bfloat16* embed_tokens = nullptr;

    // 16 layers

        // self_attn
    std::array<__nv_bfloat16*, 16> q_proj = {};
    std::array<__nv_bfloat16*, 16> k_proj = {};
    std::array<__nv_bfloat16*, 16> v_proj = {};
    std::array<__nv_bfloat16*, 16> o_proj = {};

        // ffn
    std::array<__nv_bfloat16*, 16> gate_proj = {};
    std::array<__nv_bfloat16*, 16> up_proj   = {};
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
            q_proj = other.q_proj;
            k_proj = other.k_proj;
            v_proj = other.v_proj;
            o_proj = other.o_proj;

            gate_proj = other.gate_proj;
            up_proj   = other.up_proj;
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