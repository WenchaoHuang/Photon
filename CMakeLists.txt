# Copyright (c) 2025 Wenchao Huang <physhuangwenchao@gmail.com>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

cmake_minimum_required(VERSION 3.18)

# Project name: Photon
project(Photon LANGUAGES CXX CUDA)

# Organize targets into folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Library options
option(PHOTON_BUILD_TESTS "Build tests for photon library" OFF)
option(PHOTON_BUILD_EXAMPLES "Build examples for photon library" OFF)

# Optix SDK
set(OPTIX_INCLUDE "${PROJECT_SOURCE_DIR}/deps/optix-dev/include" CACHE PATH "Path to OptiX include directory")

# Validate Optix include
if(NOT EXISTS "${OPTIX_INCLUDE}/optix.h")
	message(FATAL_ERROR "OptiX header not found in: ${OPTIX_INCLUDE}")
endif()

# Target name
set(TARGET_NAME photon)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Source files
file(GLOB_RECURSE PHOTON_SOURCES
    "${PROJECT_SOURCE_DIR}/include/photon/*.h"
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
    "${PROJECT_SOURCE_DIR}/src/*.cu"
    "${PROJECT_SOURCE_DIR}/src/*.h"
)

# Create the static library
add_library(${TARGET_NAME} STATIC ${PHOTON_SOURCES})

# Link directories
target_link_libraries(${TARGET_NAME} PRIVATE nucleus)

# Include directories
target_include_directories(${TARGET_NAME} PUBLIC ${OPTIX_INCLUDE})
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/photon)

# MSVC settings
if(MSVC)
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/MP /W4 /WX /utf-8>)
endif()

# NVCC flags
target_compile_options(${TARGET_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# CMake properties
set_target_properties(${TARGET_NAME} PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75"
)

# Filters
source_group("Source Files" FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/device_context_impl.h)

# Optional: Add tests if enabled
if(PHOTON_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Optional: Add examples if enabled
if(PHOTON_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()