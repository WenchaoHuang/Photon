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

# Target name
set(TARGET_NAME collision_detection)

# Find and configure CUDA (required for cuda_compile_ptx and bin2c)
find_package(CUDA REQUIRED)

# Source files
file(GLOB TEST_CPP_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
)
file(GLOB TEST_CUDA_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

# Create the executable target
add_executable(${TARGET_NAME} ${TEST_CPP_SOURCES} ${TEST_CUDA_SOURCES})

# List of CUDA source files that should be compiled to PTX format
set(PTX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/collision_pipeline.cu)

# Generated PTX headers directory
set(GENERATED_HEADERS_DIR ${CMAKE_CURRENT_BINARY_DIR}/ptx)

# Nucleus include directory
get_target_property(NUCLEUS_INCLUDE nucleus INTERFACE_INCLUDE_DIRECTORIES)

# PTX Generation Pipeline
foreach(PTX_SOURCE ${PTX_SOURCES})
    # Extract the base name (without extension)
    get_filename_component(base ${PTX_SOURCE} NAME_WE)

    # Compile CUDA file into PTX
    cuda_compile_ptx(${base}_ptx ${PTX_SOURCE} OPTIONS
        -I ${NUCLEUS_INCLUDE}
        -I ${OPTIX_INCLUDE}
        -std=c++20
    )

    # Generated header file path
    set(PTX_HEADER ${GENERATED_HEADERS_DIR}/${base}.ptx.h)

	# Convert PTX into a C header using bin2c
    add_custom_command(
        OUTPUT ${PTX_HEADER}
		COMMAND ${CUDA_TOOLKIT_ROOT_DIR}/bin/bin2c -p 0 -st -c -n ${base}_ptx ${${base}_ptx} > ${PTX_HEADER}
        DEPENDS ${${base}_ptx}
        COMMENT "Converting ${${base}_ptx} to ${PTX_HEADER} with bin2c"
    )

    # Add PTX source and generated *.ptx file to the target
    target_sources(${TARGET_NAME} PRIVATE ${${base}_ptx})
    target_sources(${TARGET_NAME} PRIVATE ${PTX_HEADER})

    # Group PTX file in IDE project views
    source_group("Generated PTX" FILES ${${base}_ptx})
endforeach()

# Link directories
target_link_libraries(${TARGET_NAME} PRIVATE photon nucleus)

# Make the generated header directory available for #include
target_include_directories(${TARGET_NAME} PRIVATE ${GENERATED_HEADERS_DIR})

# CUDA-Specific Properties
set_target_properties(${TARGET_NAME} PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75"
    FOLDER "Examples"
)
