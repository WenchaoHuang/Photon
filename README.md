# Photon: Modern C++20 Bindings for NVIDIA OptiX

[![License](https://img.shields.io/github/license/WenchaoHuang/Photon)](LICENSE)
[![Commit Activity](https://img.shields.io/github/commit-activity/y/WenchaoHuang/Photon/main)](https://github.com/WenchaoHuang/Photon/commits/main)

> [!WARNING]
> Photon is under active development. The API is not yet stable, and breaking changes may be introduced while the design is refined.

## Overview

Photon is a C++20 wrapper around NVIDIA OptiX. It provides RAII-managed OptiX objects, typed geometry build inputs, acceleration-structure helpers, pipeline construction, shader binding table records, and denoiser integration.

Photon builds on [Nucleus](https://github.com/WenchaoHuang/Nucleus) for CUDA devices, streams, allocators, arrays, images, and device spans. The public namespace is `pt`; Nucleus uses `ns`.

## Features

- RAII-managed OptiX device contexts, modules, program groups, pipelines, and acceleration structures
- Program factories for ray generation, miss, exception, hitgroup, and callable programs
- Geometry acceleration structures for triangles, custom AABBs, curves, and spheres
- Instance acceleration structures and OptiX instance helpers
- Typed BuildInput wrappers that infer buffer format, element count, stride, and index size
- Raw byte-span BuildInput overloads for padded, interleaved, or application-defined layouts
- Owned host-side device-address and geometry-flag arrays with safe copy semantics
- GAS build, rebuild, and refit workflows with cached BuildInput ownership
- Typed shader binding table records and pipeline launch parameters
- OptiX denoiser support, including version-dependent temporal and upscaling modes

## Requirements

- A CUDA-capable NVIDIA GPU
- CUDA Toolkit with C++20 compiler support
- NVIDIA OptiX SDK headers, supplied by the `deps/optix-dev` submodule by default
- CMake 3.18 or newer for the library
- CMake 3.27 or newer when building the included OptiX-IR examples or tests
- A C++20 compiler supported by the installed CUDA Toolkit
- Nucleus available as a CMake target named `nucleus`

OptiX feature availability follows the SDK version used to compile Photon. Curves require OptiX 7.1 or newer, and sphere primitives require OptiX 7.5 or newer.

## Getting Started

Create a workspace containing Nucleus and Photon, and clone Photon together with its OptiX header submodule:

```bash
mkdir photon-workspace
cd photon-workspace
git clone https://github.com/WenchaoHuang/Nucleus.git
git clone --recurse-submodules https://github.com/WenchaoHuang/Photon.git
```

Photon currently expects Nucleus to be added to the same CMake build before Photon. A minimal parent project can use the following layout:

```text
workspace/
├── CMakeLists.txt
├── Nucleus/
└── Photon/
```

```cmake
cmake_minimum_required(VERSION 3.18)
project(MyRayTracer LANGUAGES CXX CUDA)

add_subdirectory(Nucleus)
add_subdirectory(Photon)

add_executable(my_ray_tracer main.cpp)
target_link_libraries(my_ray_tracer PRIVATE Photon::photon)
```

Configure and build the parent project:

```bash
cmake -S . -B build
cmake --build build --config Release
```

The OptiX include directory defaults to `Photon/deps/optix-dev/include`. Override it when using another SDK location:

```bash
cmake -S . -B build -DOPTIX_INCLUDE=/path/to/OptiX/include
```

Photon provides these CMake options:

| Option | Default | Description |
| --- | --- | --- |
| `PHOTON_BUILD_SHARED_LIB` | `ON` | Build Photon as a shared library |
| `PHOTON_BUILD_EXAMPLES` | `OFF` | Build the example applications |
| `PHOTON_BUILD_TESTS` | `OFF` | Build the test executable |

The library currently targets CUDA architecture `75`. A parent project can override it after adding Photon:

```cmake
set_property(TARGET photon PROPERTY CUDA_ARCHITECTURES 86)
```

## Basic Usage

### Device context and pipeline

The following fragment matches the current module and program-group API. `ray_tracing_optixir` represents an embedded OptiX-IR byte array, such as one generated with `bin2c`.

```cpp
#include <nucleus/runtime.h>
#include <photon/device_context.h>
#include <photon/pipeline.h>

auto device = ns::Runtime::device(0);
auto context = pt::DeviceContext::create(device);
auto allocator = device->defaultAllocator();
auto& stream = device->defaultStream();

OptixPipelineCompileOptions pipelineOptions{};
pipelineOptions.pipelineLaunchParamsVariableName = "launchParams";
pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;

auto module = context->createModule(ray_tracing_optixir, pipelineOptions);

auto raygen = pt::Program::raygen(module->entry("__raygen__main"));
auto miss = pt::Program::miss(module->entry("__miss__main"));
auto hit = pt::Program::hitgroup({}, {}, module->entry("__closesthit__main"));

pt::Pipeline pipeline(context, {raygen, hit, miss}, pipelineOptions);
```

### Typed triangle build input

Typed setters infer the OptiX format, element count, and stride from the device-span element type:

```cpp
#include <nucleus/array_1d.h>
#include <photon/accel_struct.h>
#include <photon/build_inputs.h>

ns::Array<float3> vertices(allocator, vertexCount);
ns::Array<unsigned int> indices(allocator, indexCount);

pt::BuildInputTriangles buildInput;
buildInput.setVertexBuffer(vertices.span());
buildInput.setIndexBuffer(indices.span());

pt::AccelStructTriangle gas(context);
gas.build(stream, allocator, {buildInput}, {});
```

For scalar triangle indices, the element count must be divisible by three. Photon groups the values into index triplets and selects the matching OptiX index format. Signed and unsigned integer storage is accepted intentionally; signed values keep their bit pattern and are interpreted through the corresponding unsigned OptiX format.

The same typed pattern is available for the other primitive families:

```cpp
pt::BuildInputSpheres spheres;
spheres.setVertexBuffer(centerBuffer.span());
spheres.setRadiusBuffer(radiusBuffer.span());

pt::BuildInputCurves curves;
curves.setCurveType(OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE);
curves.setVertexBuffer(controlPointBuffer.span());
curves.setWidthBuffer(widthBuffer.span());
curves.setIndexBuffer(segmentIndexBuffer.span());

pt::BuildInputAabbs customPrimitives;
customPrimitives.setAabbBuffer(aabbBuffer.span());
```

### Explicit buffer layouts

Raw overloads accept device byte spans while preserving explicit format, count, stride, and element-size control. This is useful for padded or interleaved storage:

```cpp
const dev::Span<const ns::byte> motionVertexBuffers[] = {
    ns::as_bytes(vertexBuffer0.span()),
    ns::as_bytes(vertexBuffer1.span()),
};

pt::BuildInputTriangles buildInput;
buildInput.setVertexBuffers(
    motionVertexBuffers,
    pt::VertexFormat::Float3,
    vertexCount,
    vertexStrideInBytes);
```

BuildInput objects own the host-side arrays required by OptiX, including the arrays of device addresses and geometry flags. They do **not** own the referenced GPU allocations; those buffers must remain valid while the acceleration structure is built, rebuilt, or refitted.

### Geometry flags and SBT offsets

Geometry flags and `numSbtRecords` are configured together:

```cpp
buildInput.setGeometryFlags(
    OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
    materialCount);
```

Alternatively, pass one flag for each SBT record:

```cpp
const OptixGeometryFlags flags[] = {
    OPTIX_GEOMETRY_FLAG_NONE,
    OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
};

buildInput.setGeometryFlags(flags);
```

SBT index-offset buffers are configured independently and do not change the record count:

```cpp
buildInput.setSbtIndexOffsets(materialIndexBuffer.span());
```

### Native OptiX descriptors

`native()` returns a lightweight OptiX descriptor whose host pointers refer to storage owned by the BuildInput object:

```cpp
OptixBuildInputTriangleArray native = buildInput.native();
```

The returned host pointers remain valid only while the source BuildInput is alive and has not been modified. The concrete GAS classes avoid this lifetime hazard by storing copies of their BuildInput wrappers.

## Examples

- `examples/phong_scene` builds one scene containing triangle, sphere, curve, and custom-AABB GAS objects, assembles an IAS and SBT, launches a shading pipeline, and displays or writes the result.
- `examples/collision_detection` builds a large custom-primitive GAS and runs an OptiX collision-counting pipeline.

The example CMake targets compile `.cu` files as OptiX IR and convert the generated data into headers with `bin2c`.

## Project Status

Photon currently exposes the core building blocks needed by the included examples, but it is not yet a stable or complete high-level rendering framework. BuildInput validation, API consistency, and internal implementation reuse are still being refined.

## License

Photon is distributed under the terms of the [MIT License](LICENSE).
