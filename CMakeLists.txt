cmake_minimum_required(VERSION 3.22)
project(NanoInfer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

include(cmake/dependencies.cmake)

set(NANOINFER_CORE_SOURCES
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
    src/tokenizer_bridge.cpp
    src/tokenizer_bridge.h
    src/llama_weights.cpp
    src/llama_weights.h
    src/cuda_device_buffer.cpp
    src/cuda_device_buffer.h
    src/device_arena.hpp
    src/inference.cpp
    src/inference.h
    src/macro.h
    src/llama.cpp
    src/llama.h
)

add_library(nano_infer_core STATIC ${NANOINFER_CORE_SOURCES})
target_include_directories(nano_infer_core PUBLIC src)
target_link_libraries(
    nano_infer_core PUBLIC
    CUDA::cudart
    CUDA::cublas
    nlohmann_json::nlohmann_json
)

add_executable(nano_infer src/main.cpp)
target_link_libraries(nano_infer PRIVATE nano_infer_core)

add_executable(nano_infer_prompt tools/run_prompt.cpp)
target_link_libraries(nano_infer_prompt PRIVATE nano_infer_core)

include(CTest)
if(BUILD_TESTING)
    add_executable(kernel_unit_tests tests/kernel_unit_tests.cu)
    target_link_libraries(kernel_unit_tests PRIVATE nano_infer_core)
    add_test(NAME kernel_unit_tests COMMAND kernel_unit_tests)
endif()
