#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include <nlohmann/json.hpp>

#include "macro.h"
#include "llama.h"


using json = nlohmann::json;


int check_gpu_status()
{
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "No CUDA device found.\n";
        return -1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    constexpr double bytes_per_mib = double(1LL << 20);
    constexpr double bytes_per_gib = double(1LL << 30);
    std::cout << "Global memory: " << prop.totalGlobalMem / bytes_per_mib << "MB\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    size_t free_mem;
    size_t total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    std::cout << "Free memory: " << free_mem / bytes_per_gib << "GB, total memory: " << total_mem / bytes_per_gib << "GB" << std::endl;
    return 0;
}

json parse_safetensors_header(const std::vector<std::byte>& header_buffer)
{
    const char* first = reinterpret_cast<const char*>(header_buffer.data());
    const char* last = first + header_buffer.size();

    return json::parse(first, last);
}

void copy_weights_to_gpu(std::ifstream& model_file, __nv_bfloat16* device_dst, std::size_t weight_bytes)
{
    constexpr std::size_t CHUNK_BYTES = 256ull << 20;

    void* host_buffer = nullptr;
    CUDA_CHECK(cudaMallocHost(&host_buffer, CHUNK_BYTES));

    try {
        std::size_t copied = 0;
        while (copied < weight_bytes) {
            std::size_t bytes_to_copy = std::min(CHUNK_BYTES, weight_bytes - copied);

            model_file.read(reinterpret_cast<char*>(host_buffer), bytes_to_copy);
            if (static_cast<std::size_t>(model_file.gcount()) != bytes_to_copy) {
                throw std::runtime_error("unexpected EOF while reading weight data.");
            }

            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<std::byte*>(device_dst) + copied,
                host_buffer,
                bytes_to_copy,
                cudaMemcpyHostToDevice
            ));

            copied += bytes_to_copy;
        }
    } catch (...) {
        cudaFreeHost(host_buffer);
        throw;
    }

    CUDA_CHECK(cudaFreeHost(host_buffer));
}

Llama3_2 load_llama_weights(const std::string& model_path)
{
    // TODO: rearrange qkv proj weight
    std::ifstream model_file(model_path, std::ios::binary);

    if (!model_file) {
        throw std::runtime_error("safetensors not found.");
    }

    // read header
    uint64_t header_size;
    model_file.read(reinterpret_cast<char*>(&header_size), sizeof(uint64_t));
    std::vector<std::byte> header_buffer(header_size, std::byte(0));
    model_file.read(reinterpret_cast<char*>(header_buffer.data()), header_size);
    json header = parse_safetensors_header(header_buffer);

    // read weight buffer
    Llama3_2 weights = {};
    uint64_t weight_bytes = 0;
    std::vector<std::pair<__nv_bfloat16**, uint64_t>> tensor_offsets = {};

    auto register_tensor_offset =
        [&header, &weight_bytes, &tensor_offsets](const std::string& key, __nv_bfloat16** weight_slot) {
            const auto& data_offsets = header.at(key).at("data_offsets");

            tensor_offsets.emplace_back(weight_slot, data_offsets.at(0).get<uint64_t>());
            weight_bytes = std::max(weight_bytes, data_offsets.at(1).get<uint64_t>());
        };

    const auto& embedding_offsets = header.at("model.embed_tokens.weight").at("data_offsets");
    tensor_offsets.emplace_back(&weights.embed_tokens, embedding_offsets.at(0).get<uint64_t>());
    tensor_offsets.emplace_back(&weights.lm_head, embedding_offsets.at(0).get<uint64_t>());
    weight_bytes = std::max(weight_bytes, embedding_offsets.at(1).get<uint64_t>());

    register_tensor_offset("model.norm.weight", &weights.norm);

    for (int layer_idx = 0; layer_idx < 16; layer_idx++) {
        const std::string layer_prefix = "model.layers." + std::to_string(layer_idx);
        register_tensor_offset(layer_prefix + ".input_layernorm.weight", &weights.input_layernorm[layer_idx]);
        register_tensor_offset(layer_prefix + ".mlp.down_proj.weight", &weights.down_proj[layer_idx]);
        register_tensor_offset(layer_prefix + ".mlp.gate_proj.weight", &weights.gate_proj[layer_idx]);
        register_tensor_offset(layer_prefix + ".mlp.up_proj.weight", &weights.up_proj[layer_idx]);
        register_tensor_offset(layer_prefix + ".post_attention_layernorm.weight", &weights.post_attention_layernorm[layer_idx]);
        register_tensor_offset(layer_prefix + ".self_attn.k_proj.weight", &weights.k_proj[layer_idx]);
        register_tensor_offset(layer_prefix + ".self_attn.o_proj.weight", &weights.o_proj[layer_idx]);
        register_tensor_offset(layer_prefix + ".self_attn.q_proj.weight", &weights.q_proj[layer_idx]);
        register_tensor_offset(layer_prefix + ".self_attn.v_proj.weight", &weights.v_proj[layer_idx]);
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&weights.weight), weight_bytes));
    weights.weight_bytes = weight_bytes;

    copy_weights_to_gpu(model_file, weights.weight, weight_bytes);

    for (auto& tensor_offset: tensor_offsets) {
        *tensor_offset.first = reinterpret_cast<__nv_bfloat16*>(
            reinterpret_cast<std::byte*>(weights.weight) + tensor_offset.second
        );
    }

    return weights;
}

int main()
{
    if (check_gpu_status() == -1) {
        return -1;
    }
    std::cout << "\n\n\n";

    Llama3_2 llama = load_llama_weights("model.safetensors");

    return 0;
}
