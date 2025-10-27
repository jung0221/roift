cmake_minimum_required(VERSION 3.15)
project(gft)


file(GLOB_RECURSE GFT_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)

add_library(gft STATIC ${GFT_SOURCES})
target_include_directories(gft PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(ZLIB_FOUND)
    target_include_directories(gft PRIVATE ${ZLIB_INCLUDE_DIRS})
    target_link_libraries(gft PRIVATE ZLIB::ZLIB)
else()
    target_compile_definitions(gft PRIVATE ZLIB_NOT_FOUND)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(gft PRIVATE OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_definitions(gft PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(gft PRIVATE /W3)
else()
    target_compile_options(gft PRIVATE -Wall -O3)
endif()
