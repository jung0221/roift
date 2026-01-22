###############################################################################
# CMakeLists.txt for gft_delta - GPU-accelerated ROIFT with Delta-Stepping
#
# This builds the GPU version using CUDA for:
# - Delta-Stepping OIFT (parallel shortest path)
# - ORelax GPU (parallel relaxation)
###############################################################################

cmake_minimum_required(VERSION 3.18)  # CUDA support improved in 3.18+

project(gft_delta LANGUAGES CXX CUDA)

# CUDA settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Check compute capability
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Auto-detect or use common architectures
    # 52 = Maxwell (GTX 900 series)
    # 61 = Pascal (GTX 1000 series)
    # 75 = Turing (RTX 2000 series)
    # 86 = Ampere (RTX 3000 series)
    # 89 = Ada Lovelace (RTX 4000 series)
    set(CMAKE_CUDA_ARCHITECTURES "52;61;75;86;89")
endif()

message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Toolkit: ${CUDAToolkit_LIBRARY_DIR}")

###############################################################################
# GFT Library (CPU version - for non-GPU parts)
###############################################################################

# Find GFT library
set(GFT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../gft")
set(GFT_INCLUDE_DIR "${GFT_ROOT}/include")

# Check if gft library exists
if(EXISTS "${GFT_ROOT}/lib/libgft.lib")
    set(GFT_LIBRARY "${GFT_ROOT}/lib/libgft.lib")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../build/Release/gft.lib")
    set(GFT_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../build/Release/gft.lib")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../build/Debug/gft.lib")
    set(GFT_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../build/Debug/gft.lib")
else()
    message(WARNING "GFT library not found. Build the main project first.")
    # Try to find it later
    set(GFT_LIBRARY "gft")
endif()

message(STATUS "GFT Include: ${GFT_INCLUDE_DIR}")
message(STATUS "GFT Library: ${GFT_LIBRARY}")

###############################################################################
# Third-party dependencies
###############################################################################

# ZLIB linking - prefer imported target then legacy variables
# This follows the same pattern as the root CMakeLists.txt
if (TARGET ZLIB::ZLIB)
    message(STATUS "ZLIB found via imported target")
elseif (TARGET zlibstatic)
    add_library(ZLIB::ZLIB ALIAS zlibstatic)
    message(STATUS "ZLIB found as zlibstatic")
elseif (TARGET zlib)
    add_library(ZLIB::ZLIB ALIAS zlib)
    message(STATUS "ZLIB found as zlib")
elseif (DEFINED ZLIB_FOUND AND ZLIB_FOUND)
    if (DEFINED ZLIB_INCLUDE_DIRS)
        message(STATUS "ZLIB found via ZLIB_INCLUDE_DIRS")
    endif()
else()
    message(STATUS "ZLIB not found - will try to link anyway")
endif()

###############################################################################
# Delta-Stepping GPU Library
###############################################################################

# CUDA source files
set(CUDA_SOURCES
    src/delta_stepping.cu
    src/orelax_gpu.cu
    src/gaussian_blur_gpu.cu
)

# Create CUDA library
add_library(gft_delta_gpu STATIC ${CUDA_SOURCES})

target_include_directories(gft_delta_gpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${GFT_INCLUDE_DIR}
)

target_link_libraries(gft_delta_gpu PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
)

# CUDA compiler options - Release (optimized)
target_compile_options(gft_delta_gpu PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:
        --expt-relaxed-constexpr
        -O3
        --use_fast_math
        -lineinfo
    >
)

# CUDA compiler options - Debug (device debugging enabled)
target_compile_options(gft_delta_gpu PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:
        --expt-relaxed-constexpr
        -G  # Enable device debugging
        -g
    >
)

# Default for other configs
target_compile_options(gft_delta_gpu PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Release>>,$<NOT:$<CONFIG:Debug>>>:
        --expt-relaxed-constexpr
        -O2
        --use_fast_math
        -lineinfo
    >
)

###############################################################################
# Main Executable - oiftrelax_gpu
###############################################################################

add_executable(oiftrelax_gpu 
    oiftrelax_gpu.cpp
)

target_include_directories(oiftrelax_gpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${GFT_INCLUDE_DIR}
)

target_link_libraries(oiftrelax_gpu PRIVATE
    gft_delta_gpu
    ${GFT_LIBRARY}
    CUDA::cudart
)

# ZLIB linking (same as root CMakeLists.txt)
if (TARGET ZLIB::ZLIB)
    target_link_libraries(oiftrelax_gpu PRIVATE ZLIB::ZLIB)
elseif (DEFINED ZLIB_LIBRARIES)
    target_link_libraries(oiftrelax_gpu PRIVATE ${ZLIB_LIBRARIES})
endif()

if (DEFINED ZLIB_INCLUDE_DIRS)
    target_include_directories(oiftrelax_gpu PRIVATE ${ZLIB_INCLUDE_DIRS})
endif()

# Copy CUDA runtime DLLs for Windows
if(WIN32)
    add_custom_command(TARGET oiftrelax_gpu POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:CUDA::cudart>
            $<TARGET_FILE_DIR:oiftrelax_gpu>
        COMMENT "Copying CUDA runtime DLL"
    )
endif()

###############################################################################
# Benchmark executable
###############################################################################

add_executable(benchmark_gpu
    benchmark_gpu.cpp
)

target_include_directories(benchmark_gpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${GFT_INCLUDE_DIR}
)

target_link_libraries(benchmark_gpu PRIVATE
    gft_delta_gpu
    ${GFT_LIBRARY}
    CUDA::cudart
)

# ZLIB linking (same as root CMakeLists.txt)
if (TARGET ZLIB::ZLIB)
    target_link_libraries(benchmark_gpu PRIVATE ZLIB::ZLIB)
elseif (DEFINED ZLIB_LIBRARIES)
    target_link_libraries(benchmark_gpu PRIVATE ${ZLIB_LIBRARIES})
endif()

if (DEFINED ZLIB_INCLUDE_DIRS)
    target_include_directories(benchmark_gpu PRIVATE ${ZLIB_INCLUDE_DIRS})
endif()

###############################################################################
# Install rules
###############################################################################

install(TARGETS oiftrelax_gpu gft_delta_gpu
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)

###############################################################################
# Print configuration summary
###############################################################################

message(STATUS "")
message(STATUS "=== gft_delta Configuration ===")
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
