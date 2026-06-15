#pragma once


#include "llama.h"


template <typename T>
__device__ __host__ __forceinline__
T* kv_cache_layer_begin(T* kv_cache, int layer_idx)
{
    constexpr std::size_t cache_stride = 2 * K_PROJ_DIM;
    constexpr std::size_t layer_stride = static_cast<std::size_t>(MAX_TOKEN_LEN)
                                         * cache_stride;
    return kv_cache + static_cast<std::size_t>(layer_idx) * layer_stride;
}