cmake_minimum_required(VERSION 3.15)
project(ctsegmentation_src)

add_subdirectory(gft)

set(SRC_OIFT
    ${CMAKE_CURRENT_SOURCE_DIR}/oiftrelax.cpp
)

add_executable(oiftrelax ${SRC_OIFT})
target_include_directories(oiftrelax PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gft/include)
target_link_libraries(oiftrelax PRIVATE gft)

option(CT_USE_STATIC_ZLIB "Prefer static zlib (if available in C:/libs/zlib/lib)" ON)

if(ZLIB_FOUND)
    if(CT_USE_STATIC_ZLIB)
        # prefer a static library if installed (zs.lib is produced by some zlib builds)
        if(EXISTS "C:/libs/zlib/lib/zs.lib")
            target_link_libraries(oiftrelax PRIVATE "C:/libs/zlib/lib/zs.lib")
            target_include_directories(oiftrelax PRIVATE "C:/libs/zlib/include")
        elseif(EXISTS "C:/libs/zlib/lib/zlibstatic.lib")
            target_link_libraries(oiftrelax PRIVATE "C:/libs/zlib/lib/zlibstatic.lib")
            target_include_directories(oiftrelax PRIVATE "C:/libs/zlib/include")
        else()
            target_link_libraries(oiftrelax PRIVATE ZLIB::ZLIB)
        endif()
    else()
        target_link_libraries(oiftrelax PRIVATE ZLIB::ZLIB)
    endif()
endif()

if(DEFINED ZLIB_INCLUDE_DIR)
    target_include_directories(oiftrelax PRIVATE ${ZLIB_INCLUDE_DIR})
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(oiftrelax PRIVATE OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_definitions(oiftrelax PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
