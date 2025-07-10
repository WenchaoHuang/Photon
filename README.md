# Photon: Open-Source Optix C++ API

[![License](https://img.shields.io/github/license/WenchaoHuang/Photon)](LICENSE)
[![Commit Activity](https://img.shields.io/github/commit-activity/y/WenchaoHuang/Photon/main)](https://github.com/WenchaoHuang/Photon/commits/main)

> [!WARNING]
**This project is in development.** The API is unstable, features may be added or removed, and breaking changes are likely to occur frequently and without notice as the design is refined.

## Overview
Photon is a modern C++20 API wrapper for Nvidia Optix, designed to streamline GPU ray tracing development. It aims to provide type safety, modularity, and ease of integration with CUDA workflows.
Photon is part of an open-source GPU computing ecosystem along with [Nucleus](https://github.com/WenchaoHuang/Nucleus).

**Key Features:**
- Modern C++20 interface for Optix
- Type-safe resource management
- Easy device and context management
- Support for Optix pipelines, programs, acceleration structures
- Designed for extensibility and integration with CUDA projects

## Getting Started

### Prerequisites
- CUDA Toolkit (>= 11.0)
- Optix SDK (>= 7.0)
- CMake (>= 3.18)
- C++20 compatible compiler

### Build Instructions

```bash
git clone https://github.com/WenchaoHuang/Photon.git
cd Photon
mkdir build
cd build
cmake ..
```

## Usage Example
Below is a simple example showing how to initialize Photon and create core context objects:

```cpp
// Nucleus headers
#include <nucleus/device.h>
#include <nucleus/context.h>

// Photon headers
#include <photon/programs.h>
#include <photon/pipeline.h>
#include <photon/accel_struct.h>
#include <photon/device_context.h>

// Contexts
auto device = ns::Context::getInstance()->device(0);
auto devCtx = pt::DeviceContext::create(device);
auto alloc = device->defaultAllocator();
auto stream = ns::Stream(device);

// TODO: pipeline, programs, accel_struct usage...
```
More usage examples and documentation are coming soon.

## Documentation
Comprehensive documentation is under development and will cover API reference, tutorials, and best practices.

## License
Photon is distributed under the terms of the [MIT License](LICENSE).
