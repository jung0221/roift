cmake_minimum_required(VERSION 3.18)
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# CUDA kernel library (both fmax and additive)
add_library(oift_gpu_kernels STATIC oift_gpu.cu)
set_target_properties(oift_gpu_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89")
target_include_directories(oift_gpu_kernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Unified CLI binary (cost_mode selectable at runtime)
add_executable(oiftrelax_gpu oiftrelax_gpu.cpp)
target_link_libraries(oiftrelax_gpu PRIVATE oift_gpu_kernels gft CUDA::cudart_static)
target_include_directories(oiftrelax_gpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../gft/include)

# Keep old fmax binary for backward compat
add_library(oift_fmax_kernels STATIC oift_fmax_gpu.cu)
set_target_properties(oift_fmax_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89")
target_include_directories(oift_fmax_kernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_executable(oiftrelax_fmax_gpu oiftrelax_fmax_gpu.cpp)
target_link_libraries(oiftrelax_fmax_gpu PRIVATE oift_fmax_kernels gft CUDA::cudart_static)
target_include_directories(oiftrelax_fmax_gpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../gft/include)
