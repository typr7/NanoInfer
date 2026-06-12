#include <iostream>

#include <cuda_runtime.h>

#include "macro.h"
#include "llama_weights.h"


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

int main()
{
    if (check_gpu_status() == -1) {
        return -1;
    }
    std::cout << "\n\n\n";

    Llama3_2 llama = load_llama_weights("model.safetensors");

    return 0;
}
