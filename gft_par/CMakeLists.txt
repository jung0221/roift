# CMakeLists.txt para GFT Parallel com OpenMP

cmake_minimum_required(VERSION 3.15)
project(gft_parallel)

# Habilitar OpenMP
find_package(OpenMP REQUIRED)

# Configurar C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Adicionar flags de otimização
if(MSVC)
    # Visual Studio flags
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /openmp")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /openmp")
else()
    # GCC/Clang flags
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -fopenmp")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -fopenmp")
endif()

# Adicionar diretórios de include do projeto original
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../gft/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../gft/src
)

# Pegar todas as sources do gft original
file(GLOB_RECURSE GFT_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/../gft/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../gft/src/*.c"
)

# Remover gft_ift.cpp da lista (vamos usar a versão paralela)
list(FILTER GFT_SOURCES EXCLUDE REGEX "gft_ift\\.cpp$")

# Criar biblioteca paralela com todas as sources do gft EXCETO gft_ift.cpp
# MAIS a versão paralela gft_ift_parallel.cpp
add_library(gft_parallel STATIC
    ${GFT_SOURCES}
    src/gft_ift_parallel.cpp
)

# Linkar OpenMP e ZLIB
target_link_libraries(gft_parallel PRIVATE OpenMP::OpenMP_CXX)

# Linkar ZLIB (mesmo logic do projeto principal)
if (TARGET ZLIB::ZLIB)
    target_link_libraries(gft_parallel PRIVATE ZLIB::ZLIB)
elseif (TARGET zlibstatic)
    add_library(ZLIB::ZLIB ALIAS zlibstatic)
    target_link_libraries(gft_parallel PRIVATE ZLIB::ZLIB)
elseif (TARGET zlib)
    add_library(ZLIB::ZLIB ALIAS zlib)
    target_link_libraries(gft_parallel PRIVATE ZLIB::ZLIB)
elseif (DEFINED ZLIB_FOUND AND ZLIB_FOUND)
    if (DEFINED ZLIB_INCLUDE_DIRS)
        target_include_directories(gft_parallel PRIVATE ${ZLIB_INCLUDE_DIRS})
    endif()
    if (DEFINED ZLIB_LIBRARIES)
        target_link_libraries(gft_parallel PRIVATE ${ZLIB_LIBRARIES})
    endif()
else()
    target_compile_definitions(gft_parallel PRIVATE ZLIB_NOT_FOUND)
endif()

# Criar executável oiftrelax_parallel
set(OIFTRELAX_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../oiftrelax.cpp")
if(EXISTS "${OIFTRELAX_SRC}")
    add_executable(oiftrelax_parallel "${OIFTRELAX_SRC}")
    
    # Linkar com a biblioteca paralela
    target_link_libraries(oiftrelax_parallel PRIVATE gft_parallel)
    
    # Include directories
    target_include_directories(oiftrelax_parallel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../gft/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../gft/src
    )
    
    # ZLIB linking
    if (TARGET ZLIB::ZLIB)
        target_link_libraries(oiftrelax_parallel PRIVATE ZLIB::ZLIB)
    elseif (TARGET zlib)
        add_library(ZLIB::ZLIB ALIAS zlib)
        target_link_libraries(oiftrelax_parallel PRIVATE ZLIB::ZLIB)
    elseif (DEFINED ZLIB_LIBRARIES)
        target_link_libraries(oiftrelax_parallel PRIVATE ${ZLIB_LIBRARIES})
    else()
        target_compile_definitions(oiftrelax_parallel PRIVATE ZLIB_NOT_FOUND)
    endif()
    
    # OpenMP link
    if(OpenMP_CXX_FOUND)
        target_link_libraries(oiftrelax_parallel PRIVATE OpenMP::OpenMP_CXX)
    endif()
    
    # DLL copy command (Windows)
    if (WIN32)
        set(_vcpkg_root "${CMAKE_SOURCE_DIR}/vcpkg_installed")
        set(_vcpkg_bin_release "${_vcpkg_root}/x64-windows/bin")
        set(_vcpkg_bin_debug "${_vcpkg_root}/x64-windows/debug/bin")
        set(_zlib_dll_debug "${_vcpkg_bin_debug}/zlibd1.dll")
        set(_zlib_dll_release "${_vcpkg_bin_release}/zlib1.dll")
        
        if(EXISTS "${_zlib_dll_debug}" OR EXISTS "${_zlib_dll_release}")
            add_custom_command(TARGET oiftrelax_parallel POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E echo "Copying zlib1.dll to oiftrelax_parallel folder"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "$<$<CONFIG:Debug>:${_zlib_dll_debug}>$<$<NOT:$<CONFIG:Debug>>:${_zlib_dll_release}>"
                    "$<TARGET_FILE_DIR:oiftrelax_parallel>/zlib1.dll"
                COMMENT "Copy zlib1.dll to oiftrelax_parallel target folder"
            )
        endif()
    endif()
    
    message(STATUS "Building oiftrelax_parallel with OpenMP optimization")
else()
    message(STATUS "oiftrelax source not found at ${OIFTRELAX_SRC}")
endif()

# Configurações de OpenMP recomendadas em runtime:
# - Windows: set OMP_NUM_THREADS=8
# - Linux: export OMP_NUM_THREADS=8
# Use número de cores físicos, não lógicos (sem hyperthreading)
