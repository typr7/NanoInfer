cmake_minimum_required(VERSION 3.22)
project(tvllm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

include(cmake/dependencies.cmake)

add_executable(tvllm
    src/kernels/grouped_query_attention.cu
    src/kernels/grouped_query_attention.h
    src/kernels/kv_cache.cu
    src/kernels/kv_cache.h
    src/kernels/greedy_decoding.cu
    src/kernels/greedy_decoding.h
    src/kernels/attention_softmax.cu
    src/kernels/attention_softmax.h
    src/kernels/residual_connection.cu
    src/kernels/residual_connection.h
    src/kernels/rms_norm.cu
    src/kernels/rms_norm.h
    src/kernels/rope.cu
    src/kernels/rope.h
    src/kernels/swiglu.cu
    src/kernels/swiglu.h
    src/kernels/token_embedding.cu
    src/kernels/token_embedding.h
    src/helper.hpp
    src/llama_weights.cpp
    src/llama_weights.h
    src/cuda_device_buffer.cpp
    src/cuda_device_buffer.h
    src/device_arena.hpp
    src/main.cpp
    src/inference.cpp
    src/inference.h
    src/macro.h
    src/llama.cpp
    src/llama.h
)

target_link_libraries(tvllm PRIVATE CUDA::cudart)
target_link_libraries(
    tvllm PRIVATE
    CUDA::cudart
    CUDA::cublas
    nlohmann_json::nlohmann_json
)
