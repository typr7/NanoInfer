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

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            throw std::runtime_error(std::string("CUDA error: ") +         \
                                     cudaGetErrorString(err__));           \
        }                                                                  \
    } while (0)

using json = nlohmann::json;


int checkGPUStatus()
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
    std::cout << "Global memory: " << prop.totalGlobalMem / B_TO_MB << "MB\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    size_t free_mem;
    size_t total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    std::cout << "Free memory: " << free_mem / B_TO_GB << "GB, total memory: " << total_mem / B_TO_GB << "GB" << std::endl;
    return 0;
}

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

json loadSafetensorsHeader(const std::vector<std::byte>& buffer)
{
    const char* first = reinterpret_cast<const char*>(buffer.data());
    const char* last = first + buffer.size();

    return json::parse(first, last);
}

void loadWeightToGPU(std::ifstream& file, __nv_bfloat16* dst, std::size_t weight_bytes)
{
    constexpr std::size_t CHUNK_BYTES = 256ull << 20;

    void* host_buffer = nullptr;
    CUDA_CHECK(cudaMallocHost(&host_buffer, CHUNK_BYTES));

    try {
        std::size_t copied = 0;
        while (copied < weight_bytes) {
            std::size_t bytes_to_copy = std::min(CHUNK_BYTES, weight_bytes - copied);

            file.read(reinterpret_cast<char*>(host_buffer), bytes_to_copy);
            if (static_cast<std::size_t>(file.gcount()) != bytes_to_copy) {
                throw std::runtime_error("unexpected EOF while reading weight data.");
            }

            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<std::byte*>(dst) + copied,
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
    Llama3_2 ret = {};
    uint64_t max_len = 0;
    std::vector<std::pair<__nv_bfloat16**, uint64_t>> offsets = {};

    auto aux = [&header, &max_len, &offsets](const std::string& key, __nv_bfloat16** pptr) {
        const auto& o = header.at(key).at("data_offsets");

        offsets.emplace_back(pptr, o.at(0).get<uint64_t>());
        max_len = std::max(max_len, o.at(1).get<uint64_t>());
    };

    const auto& tmp = header.at("model.embed_tokens.weight").at("data_offsets");
    offsets.emplace_back(&ret.embed_tokens, tmp.at(0).get<uint64_t>());
    offsets.emplace_back(&ret.lm_head, tmp.at(0).get<uint64_t>());
    max_len = std::max(max_len, tmp.at(1).get<uint64_t>());

    aux("model.norm.weight", &ret.norm);

    for (int i = 0; i < 16; i++) {
        aux("model.layers." + std::to_string(i) + ".input_layernorm.weight", &ret.input_layernorm[i]);
        aux("model.layers." + std::to_string(i) + ".mlp.down_proj.weight", &ret.down_proj[i]);
        aux("model.layers." + std::to_string(i) + ".mlp.gate_proj.weight", &ret.gate_proj[i]);
        aux("model.layers." + std::to_string(i) + ".mlp.up_proj.weight", &ret.up_proj[i]);
        aux("model.layers." + std::to_string(i) + ".post_attention_layernorm.weight", &ret.post_attention_layernorm[i]);
        aux("model.layers." + std::to_string(i) + ".self_attn.k_proj.weight", &ret.k_proj[i]);
        aux("model.layers." + std::to_string(i) + ".self_attn.o_proj.weight", &ret.o_proj[i]);
        aux("model.layers." + std::to_string(i) + ".self_attn.q_proj.weight", &ret.q_proj[i]);
        aux("model.layers." + std::to_string(i) + ".self_attn.v_proj.weight", &ret.v_proj[i]);
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ret.weight), max_len));
    ret.weight_bytes = max_len;

    loadWeightToGPU(safetensors_file, ret.weight, max_len);

    for (auto& p: offsets) {
        *p.first = reinterpret_cast<__nv_bfloat16*>(
            reinterpret_cast<std::byte*>(ret.weight) + p.second
        );
    }

    return ret;
}

int main()
{
    if(checkGPUStatus() == -1) {
        return -1;
    }
    std::cout << "\n\n\n";

    Llama3_2 llama = loadLlamaWeight("model.safetensors");

    return 0;
}