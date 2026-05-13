#include <iostream>
#include <fstream>
#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <queue>
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include <nlohmann/json.hpp>

#define B_TO_MB double(1LL << 20)
#define B_TO_GB double(1LL << 30)


using json = nlohmann::json;


int checkGPUStatus()
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA device found.\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Global memory: " << prop.totalGlobalMem / B_TO_MB << "MB\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    size_t free_mem;
    size_t total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Free memory: " << free_mem / B_TO_GB << "GB, total memory: " << total_mem / B_TO_GB << "GB" << std::endl;
    return 0;
}

/*
json readSafetensorsHeader(const std::string& path)
{
    std::ifstream safetensors(path, std::ios::binary);
    if (!safetensors) {
        std::cerr << "File not found.\n";
        exit(-1);
    }

    uint64_t header_size;
    safetensors.read(reinterpret_cast<char*>(&header_size), 8);
    
    std::string header(header_size, '\0');
    safetensors.read(header.data(), header_size);

    if (header.empty() || header[0] != '{') {
        std::cerr << "Invalid safetensors header.\n";
        exit(-1);
    }

    return json::parse(header);
}
*/

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
    __nv_bfloat16* weight = nullptr;
};

Llama3_2 load_llama_weight(const json& header)
{
    Llama3_2 ret = {};

    std::pair<std::string, __nv_bfloat16**> layers[] = {
        {".input_layernorm.weight", ret.input_layernorm},
        {".mlp.down_proj.weight", ret.down_proj},
        {".mlp.gate_proj.weight", ret.gate_proj},
        {".mlp.up_proj.weight", ret.up_proj},
        {".post_attention_layernorm.weight", ret.post_attention_layernorm},
        {".self_attn.k_proj.weight", ret.k_proj},
        {".self_attn.o_proj.weight", ret.o_proj},
        {".self_attn.q_proj.weight", ret.q_proj},
        {".self_attn.v_proj.weight", ret.v_proj}
    };

    cudaMalloc(&ret.weight, Llama3_2::weight_size);

    ret.embed_tokens = ret.lm_head = reinterpret_cast<__nv_bfloat16*>(
        reinterpret_cast<uint8_t*>(ret.weight)
        + (std::size_t) header.at("model.embed_tokens.weight").at("data_offsets").at(0)
    );

    ret.norm = reinterpret_cast<__nv_bfloat16*>(
        reinterpret_cast<uint8_t*>(ret.weight)
        + (std::size_t) header.at("model.norm.weight").at("data_offsets").at(0)
    );

    // 16 layers
    for (int i = 0; i < 16; i++) {
        for (auto& [suffix, pptr]: layers) {
            std::string key = "model.layers." + std::to_string(i) + suffix;
            pptr[i] = reinterpret_cast<__nv_bfloat16*>(
                reinterpret_cast<uint8_t*>(ret.weight)
                + (std::size_t) header.at(key).at("data_offsets").at(0)
            );
        }
    }

    return ret;
}



json loadSafetensorsHeader(const std::vector<std::byte>& buffer)
{
    const char* first = reinterpret_cast<const char*>(buffer.data());
    const char* last = first + buffer.size();

    return json::parse(first, last);
}

Llama3_2 loadLlamaWeight(const std::string& safetensors_path)
{
    std::ifstream safetensors_file(safetensors_path, std::ios::binary);

    if (!safetensors_file) {
        throw std::runtime_error("safetensors not found.");
    }

    // read header
    uint64_t header_size;
    safetensors_file.read(reinterpret_cast<char*>(&header_size), sizeof(uint64_t));
    std::vector<std::byte> header_buffer(header_size, std::byte(0));
    safetensors_file.read(reinterpret_cast<char*>(header_buffer.data()), header_size);
    json header = loadSafetensorsHeader(header_buffer);

    // read weight buffer

}

int main()
{
    checkGPUStatus();
    std::cout << "\n\n\n";

    json js = readSafetensorsHeader("model.safetensors");

    Llama3_2 ret = load_llama_weight(js);

    return 0;
}