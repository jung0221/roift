cmake_minimum_required(VERSION 3.15)
project(oift_experiments)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GFT include dirs (relative to this subdirectory: ../gft/include and ../gft/src)
set(GFT_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../gft/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../gft/src
)

find_package(ZLIB REQUIRED)

# Helper macro to build one experiment executable
macro(add_experiment EXP_NAME SRC_FILE)
    add_executable(${EXP_NAME} ${SRC_FILE})
    target_include_directories(${EXP_NAME} PRIVATE ${GFT_INCLUDE_DIRS})
    target_link_libraries(${EXP_NAME} PRIVATE gft)

    if (TARGET ZLIB::ZLIB)
        target_link_libraries(${EXP_NAME} PRIVATE ZLIB::ZLIB)
    elseif (DEFINED ZLIB_LIBRARIES)
        target_link_libraries(${EXP_NAME} PRIVATE ${ZLIB_LIBRARIES})
    endif()

    # Link math lib on Linux
    if(UNIX AND NOT APPLE)
        target_link_libraries(${EXP_NAME} PRIVATE m)
    endif()

    target_compile_options(${EXP_NAME} PRIVATE -O2)
endmacro()

add_experiment(exp_gradient_weight   ${CMAKE_CURRENT_SOURCE_DIR}/exp_gradient_weight.cpp)
add_experiment(exp_gaussian_rbf_relax ${CMAKE_CURRENT_SOURCE_DIR}/exp_gaussian_rbf_relax.cpp)
add_experiment(exp_coarse_to_fine    ${CMAKE_CURRENT_SOURCE_DIR}/exp_coarse_to_fine.cpp)
add_experiment(exp_26connect         ${CMAKE_CURRENT_SOURCE_DIR}/exp_26connect.cpp)
add_experiment(exp_geodesic_seeds    ${CMAKE_CURRENT_SOURCE_DIR}/exp_geodesic_seeds.cpp)

message(STATUS "Configured OIFT experiments: exp_gradient_weight, exp_gaussian_rbf_relax, exp_coarse_to_fine, exp_26connect, exp_geodesic_seeds")
