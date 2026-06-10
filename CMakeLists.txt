cmake_minimum_required(VERSION 3.22)
project(tvllm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

include(cmake/dependencies.cmake)

add_executable(tvllm
    src/swiglu.cu
    src/swiglu.h
    src/rope.cu
    src/rope.h
    src/residual_connection.cu
    src/residual_connection.h
    src/softmax.cu
    src/softmax.h
    src/grouped_query_attention.cpp
    src/grouped_query_attention.h
    src/device_arena.hpp
    src/kernels.cu
    src/kernels.h
    src/main.cpp
    src/inference.cpp
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