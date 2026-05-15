#pragma once

#include <string>
#include <stdexcept>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            throw std::runtime_error(std::string("CUDA error: ") +         \
                                     cudaGetErrorString(err__));           \
        }                                                                  \
    } while (0)