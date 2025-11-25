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

# Enales CUDA_OPTIX_COMPILATION
cmake_minimum_required(VERSION 3.27)

# Target / Object name
set(TARGET_NAME photon-tests)
set(OBJECT_NAME ${TARGET_NAME}-optix)

# Source files
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
)

# List of CUDA source files that should be compiled to optix format
set(OPTIX_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/rt_program.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/payload_test.cu"
)

# Create the executable and object target
add_executable(${TARGET_NAME} ${TEST_SOURCES})
add_library(${OBJECT_NAME} OBJECT ${OPTIX_SOURCES})
add_dependencies(${TARGET_NAME} ${OBJECT_NAME})

# List to track generated header files (populated in the loop below)
set(GENERATED_HEADERS_DIR "${CMAKE_CURRENT_BINARY_DIR}/optixir")

# Optix Generation Pipeline
foreach(OPTIX_SOURCE ${OPTIX_SOURCES})
    # Extract the base name (without extension)
    get_filename_component(base ${OPTIX_SOURCE} NAME_WE)

    # Generated header file path
    set(OPTIXIR_HEADER "${GENERATED_HEADERS_DIR}/${base}.optixir.h")

	# Convert *.optixir into a C header using bin2c
    add_custom_command(
        OUTPUT ${OPTIXIR_HEADER}
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${OBJECT_NAME}.dir/$<CONFIG>/${base}.optixir
		COMMAND bin2c.exe -st -c -n ${base}_optixir ${CMAKE_CURRENT_BINARY_DIR}/${OBJECT_NAME}.dir/$<CONFIG>/${base}.optixir > ${OPTIXIR_HEADER}
        COMMENT "Converting ${CMAKE_CURRENT_BINARY_DIR}/${OBJECT_NAME}.dir/$<CONFIG>/${base}.optixir to ${OPTIXIR_HEADER} with bin2c"
    )

    # Add generated *.optixir.h file to the target
    target_sources(${TARGET_NAME} PRIVATE ${OPTIXIR_HEADER})

    # Group generated files in IDE project views
    source_group("Generated Files" FILES ${OPTIXIR_HEADER})
endforeach()

# Link directories
target_link_libraries(${TARGET_NAME} PRIVATE photon)
target_link_libraries(${OBJECT_NAME} PRIVATE photon)

# Make the generated header directory available for #include
target_include_directories(${TARGET_NAME} PRIVATE ${GENERATED_HEADERS_DIR})

if(MSVC)
    # MSVC-specific compiler flags
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/MP /W3 /WX /utf-8>)
    target_compile_options(${OBJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/MP /W3 /WX /utf-8>)

    # For suppressing MSVC IntelliSense errors.
    add_dummy_cpp(${OBJECT_NAME})
endif()

# CUDA-Specific Properties
set_target_properties(${OBJECT_NAME} PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_OPTIX_COMPILATION ON
    CUDA_ARCHITECTURES "75"
    FOLDER "Tests"
)
set_target_properties(${TARGET_NAME} PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "75"
    FOLDER "Tests"
)