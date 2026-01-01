cmake_minimum_required(VERSION 3.15)
project(gft)

file(GLOB_RECURSE GFT_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/gft/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/gft/src/*.c"
)

add_library(gft STATIC ${GFT_SOURCES})

target_include_directories(gft
    PUBLIC BEFORE
        ${CMAKE_CURRENT_SOURCE_DIR}/gft/include
        ${CMAKE_CURRENT_SOURCE_DIR}/gft/src
)

if (TARGET ZLIB::ZLIB)
    target_link_libraries(gft PRIVATE ZLIB::ZLIB)
elseif (TARGET zlibstatic)
    add_library(ZLIB::ZLIB ALIAS zlibstatic)
    target_link_libraries(gft PRIVATE ZLIB::ZLIB)
elseif (TARGET zlib)
    add_library(ZLIB::ZLIB ALIAS zlib)
    target_link_libraries(gft PRIVATE ZLIB::ZLIB)
elseif (DEFINED ZLIB_FOUND AND ZLIB_FOUND)
    if (DEFINED ZLIB_INCLUDE_DIRS)
        target_include_directories(gft PRIVATE ${ZLIB_INCLUDE_DIRS})
    endif()
    if (DEFINED ZLIB_LIBRARIES)
        target_link_libraries(gft PRIVATE ${ZLIB_LIBRARIES})
    endif()
else()
    target_compile_definitions(gft PRIVATE ZLIB_NOT_FOUND)
endif()


if(OpenMP_CXX_FOUND)
    target_link_libraries(gft PRIVATE OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_options(gft PRIVATE
        "/I${CMAKE_CURRENT_SOURCE_DIR}/gft/include"
        "/I${CMAKE_CURRENT_SOURCE_DIR}/gft/src"
    )
else()
    target_compile_options(gft PRIVATE
        "-I${CMAKE_CURRENT_SOURCE_DIR}/gft/include"
        "-I${CMAKE_CURRENT_SOURCE_DIR}/gft/src"
    )
endif()

# oiftrelax executable (source lives at roift/oiftrelax.cpp)
set(OIFTRELAX_SRC "${CMAKE_CURRENT_SOURCE_DIR}/oiftrelax.cpp")
if(EXISTS "${OIFTRELAX_SRC}")
    add_executable(oiftrelax "${OIFTRELAX_SRC}")

    # include the gft headers so oiftrelax builds
    target_include_directories(oiftrelax PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/gft/include
        ${CMAKE_CURRENT_SOURCE_DIR}/gft/src
    )

    # link against the library produced here
    target_link_libraries(oiftrelax PRIVATE gft)

    # zlib linking: prefer imported target then legacy variables
    if (TARGET ZLIB::ZLIB)
        target_link_libraries(oiftrelax PRIVATE ZLIB::ZLIB)
    elseif (TARGET zlib)
        add_library(ZLIB::ZLIB ALIAS zlib)
        target_link_libraries(oiftrelax PRIVATE ZLIB::ZLIB)
    elseif (DEFINED ZLIB_LIBRARIES)
        target_link_libraries(oiftrelax PRIVATE ${ZLIB_LIBRARIES})
    else()
        # no zlib found — keep building but define a macro so code can handle absence
        target_compile_definitions(oiftrelax PRIVATE ZLIB_NOT_FOUND)
    endif()

    if (WIN32)
        set(_vcpkg_root "${CMAKE_SOURCE_DIR}/vcpkg_installed")
        set(_vcpkg_bin_release "${_vcpkg_root}/x64-windows/bin")
        set(_vcpkg_bin_debug "${_vcpkg_root}/x64-windows/debug/bin")
        # Copying a single DLL avoids copying many unrelated runtime files.
        add_custom_command(TARGET oiftrelax POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "Copying zlib1.dll to oiftrelax folder"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<$<CONFIG:Debug>:${_vcpkg_bin_debug}/zlib1.dll>$<$<NOT:$<CONFIG:Debug>>:${_vcpkg_bin_release}/zlib1.dll>"
                "$<TARGET_FILE_DIR:oiftrelax>/zlib1.dll"
            COMMENT "Copy zlib1.dll to oiftrelax target folder"
        )
    endif()

    # Do NOT link ITK for oiftrelax (this tool depends only on zlib + gft)
else()
    message(STATUS "oiftrelax source not found at ${OIFTRELAX_SRC}; not building oiftrelax")
endif()