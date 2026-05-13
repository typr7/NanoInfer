cmake_minimum_required(VERSION 3.22)
project(tvllm LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

include(cmake/dependencies.cmake)

add_executable(tvllm main.cpp)

target_link_libraries(tvllm PRIVATE CUDA::cudart)
target_link_libraries(
    tvllm PRIVATE
    CUDA::cudart
    CUDA::cublas
    nlohmann_json::nlohmann_json
)