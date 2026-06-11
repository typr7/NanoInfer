#include "llama_weights.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include <nlohmann/json.hpp>

#include "macro.h"


namespace {

using json = nlohmann::json;

constexpr int LAYER_COUNT = 16;
constexpr std::size_t CHUNK_BYTES = 256ull << 20;
constexpr std::size_t WEIGHT_ALIGNMENT = 256;

struct TensorSpan
{
    std::uint64_t begin = 0;
    std::uint64_t end = 0;

    std::uint64_t size() const
    {
        return end - begin;
    }
};

struct TensorCopy
{
    TensorSpan src = {};
    std::uint64_t dst_offset = 0;
};

struct PointerPatch
{
    __nv_bfloat16** slot = nullptr;
    std::uint64_t offset = 0;
};

json parse_safetensors_header(const std::vector<std::byte>& header_buffer)
{
    const char* first = reinterpret_cast<const char*>(header_buffer.data());
    const char* last = first + header_buffer.size();
    return json::parse(first, last);
}

TensorSpan get_tensor_span(const json& header, const std::string& key)
{
    const auto& data_offsets = header.at(key).at("data_offsets");
    const std::uint64_t begin = data_offsets.at(0).get<std::uint64_t>();
    const std::uint64_t end = data_offsets.at(1).get<std::uint64_t>();
    if (end < begin) {
        throw std::runtime_error("invalid safetensors offset range: " + key);
    }
    return {begin, end};
}

std::uint64_t align_offset(std::uint64_t offset)
{
    return (offset + WEIGHT_ALIGNMENT - 1) & ~(WEIGHT_ALIGNMENT - 1);
}

std::uint64_t reserve_weight_region(
    std::uint64_t& weight_bytes,
    std::uint64_t bytes
) {
    const std::uint64_t offset = align_offset(weight_bytes);
    weight_bytes = offset + bytes;
    return offset;
}

void register_tensor_copy(
    const json& header,
    std::uint64_t& weight_bytes,
    std::vector<TensorCopy>& tensor_copies,
    std::vector<PointerPatch>& pointer_patches,
    const std::string& key,
    __nv_bfloat16** weight_slot
) {
    const TensorSpan span = get_tensor_span(header, key);
    const std::uint64_t dst_offset = reserve_weight_region(weight_bytes, span.size());
    tensor_copies.push_back({span, dst_offset});
    pointer_patches.push_back({weight_slot, dst_offset});
}

void copy_tensor_to_gpu(
    std::ifstream& model_file,
    std::uint64_t data_begin,
    const TensorSpan& src,
    std::byte* device_dst,
    void* host_buffer
) {
    model_file.seekg(data_begin + src.begin, std::ios::beg);
    if (!model_file) {
        throw std::runtime_error("failed to seek safetensors data.");
    }

    std::uint64_t copied = 0;
    while (copied < src.size()) {
        const std::size_t bytes_to_copy = static_cast<std::size_t>(
            std::min<std::uint64_t>(CHUNK_BYTES, src.size() - copied)
        );

        model_file.read(reinterpret_cast<char*>(host_buffer), bytes_to_copy);
        if (static_cast<std::size_t>(model_file.gcount()) != bytes_to_copy) {
            throw std::runtime_error("unexpected EOF while reading weight data.");
        }

        CUDA_CHECK(cudaMemcpy(
            device_dst + copied,
            host_buffer,
            bytes_to_copy,
            cudaMemcpyHostToDevice
        ));

        copied += bytes_to_copy;
    }
}

void register_qkv_copy(
    const json& header,
    std::uint64_t& weight_bytes,
    std::vector<TensorCopy>& tensor_copies,
    std::vector<PointerPatch>& pointer_patches,
    const std::string& layer_prefix,
    __nv_bfloat16** weight_slot
) {
    const TensorSpan q_span = get_tensor_span(header, layer_prefix + ".self_attn.q_proj.weight");
    const TensorSpan k_span = get_tensor_span(header, layer_prefix + ".self_attn.k_proj.weight");
    const TensorSpan v_span = get_tensor_span(header, layer_prefix + ".self_attn.v_proj.weight");
    const std::uint64_t qkv_bytes = q_span.size() + k_span.size() + v_span.size();
    const std::uint64_t qkv_offset = reserve_weight_region(weight_bytes, qkv_bytes);

    tensor_copies.push_back({q_span, qkv_offset});
    tensor_copies.push_back({k_span, qkv_offset + q_span.size()});
    tensor_copies.push_back({v_span, qkv_offset + q_span.size() + k_span.size()});
    pointer_patches.push_back({weight_slot, qkv_offset});
}

void register_gate_up_copy(
    const json& header,
    std::uint64_t& weight_bytes,
    std::vector<TensorCopy>& tensor_copies,
    std::vector<PointerPatch>& pointer_patches,
    const std::string& layer_prefix,
    __nv_bfloat16** weight_slot
) {
    const TensorSpan gate_span = get_tensor_span(header, layer_prefix + ".mlp.gate_proj.weight");
    const TensorSpan up_span = get_tensor_span(header, layer_prefix + ".mlp.up_proj.weight");
    const std::uint64_t dst_offset = reserve_weight_region(weight_bytes, gate_span.size() + up_span.size());
    tensor_copies.push_back({gate_span, dst_offset});
    tensor_copies.push_back({up_span, dst_offset + gate_span.size()});
    pointer_patches.push_back({weight_slot, dst_offset});
}

void resolve_weight_pointer(__nv_bfloat16* weight_base, const PointerPatch& pointer_patch)
{
    *pointer_patch.slot = reinterpret_cast<__nv_bfloat16*>(
        reinterpret_cast<std::byte*>(weight_base) + pointer_patch.offset
    );
}

} // namespace

Llama3_2 load_llama_weights(const std::string& model_path)
{
    std::ifstream model_file(model_path, std::ios::binary);
    if (!model_file) {
        throw std::runtime_error("safetensors not found.");
    }

    std::uint64_t header_size = 0;
    model_file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    if (!model_file) {
        throw std::runtime_error("failed to read safetensors header size.");
    }

    std::vector<std::byte> header_buffer(header_size, std::byte(0));
    model_file.read(reinterpret_cast<char*>(header_buffer.data()), header_size);
    if (static_cast<std::uint64_t>(model_file.gcount()) != header_size) {
        throw std::runtime_error("failed to read safetensors header.");
    }

    const std::uint64_t data_begin = sizeof(header_size) + header_size;
    const json header = parse_safetensors_header(header_buffer);

    Llama3_2 weights = {};
    std::uint64_t weight_bytes = 0;
    std::vector<TensorCopy> tensor_copies = {};
    std::vector<PointerPatch> pointer_patches = {};

    const TensorSpan embedding_span = get_tensor_span(header, "model.embed_tokens.weight");
    const std::uint64_t embedding_offset = reserve_weight_region(weight_bytes, embedding_span.size());
    tensor_copies.push_back({embedding_span, embedding_offset});
    pointer_patches.push_back({&weights.embed_tokens, embedding_offset});
    pointer_patches.push_back({&weights.lm_head, embedding_offset});

    register_tensor_copy(
        header,
        weight_bytes,
        tensor_copies,
        pointer_patches,
        "model.norm.weight",
        &weights.norm
    );

    for (int layer_idx = 0; layer_idx < LAYER_COUNT; layer_idx++) {
        const std::string layer_prefix = "model.layers." + std::to_string(layer_idx);
        register_tensor_copy(
            header,
            weight_bytes,
            tensor_copies,
            pointer_patches,
            layer_prefix + ".input_layernorm.weight",
            &weights.input_layernorm[layer_idx]
        );
        register_tensor_copy(
            header,
            weight_bytes,
            tensor_copies,
            pointer_patches,
            layer_prefix + ".mlp.down_proj.weight",
            &weights.down_proj[layer_idx]
        );
        register_gate_up_copy(
            header,
            weight_bytes,
            tensor_copies,
            pointer_patches,
            layer_prefix,
            &weights.gate_up_proj[layer_idx]
        );
        register_tensor_copy(
            header,
            weight_bytes,
            tensor_copies,
            pointer_patches,
            layer_prefix + ".post_attention_layernorm.weight",
            &weights.post_attention_layernorm[layer_idx]
        );
        register_tensor_copy(
            header,
            weight_bytes,
            tensor_copies,
            pointer_patches,
            layer_prefix + ".self_attn.o_proj.weight",
            &weights.o_proj[layer_idx]
        );
        register_qkv_copy(
            header,
            weight_bytes,
            tensor_copies,
            pointer_patches,
            layer_prefix,
            &weights.qkv_proj[layer_idx]
        );
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&weights.weight), weight_bytes));
    weights.weight_bytes = weight_bytes;

    void* host_buffer = nullptr;
    CUDA_CHECK(cudaMallocHost(&host_buffer, CHUNK_BYTES));

    try {
        for (const TensorCopy& tensor_copy: tensor_copies) {
            copy_tensor_to_gpu(
                model_file,
                data_begin,
                tensor_copy.src,
                reinterpret_cast<std::byte*>(weights.weight) + tensor_copy.dst_offset,
                host_buffer
            );
        }
    } catch (...) {
        cudaFreeHost(host_buffer);
        cudaFree(weights.weight);
        weights.weight = nullptr;
        weights.weight_bytes = 0;
        throw;
    }

    CUDA_CHECK(cudaFreeHost(host_buffer));

    for (const PointerPatch& pointer_patch: pointer_patches) {
        resolve_weight_pointer(weights.weight, pointer_patch);
    }

    return weights;
}
