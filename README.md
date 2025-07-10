# Photon: C++ Bindings for Nvidia Optix

[![License](https://img.shields.io/github/license/WenchaoHuang/Photon)](https://opensource.org/license/mit-0)
![GitHub commit activity](https://img.shields.io/github/commit-activity/y/WenchaoHuang/Photon/main)

> [!WARNING]
**This project is in development.** The API is unstable, features may be added or removed, and breaking changes are likely to occur frequently and without notice as the design is refined.

## Overview
Comming soon ...

## Installation

### Prerequisites

- CUDA Toolkit (version 11.0 or above)
- Optix SDK (version 7.0 or above)
- CMake (version 3.18 or above)
- C++20 compatible compiler

### Build Instructions

```bash
git clone https://github.com/WenchaoHuang/Photon.git
cd Photon
mkdir build
cd build
cmake ..
```

## Documentation
Comming soon ...

```cpp
//  nucleus' headers
#include <nucleus/device.h>
#include <nucleus/context.h>

//  photon's headers
#include <photon/programs.h>
#include <photon/pipeline.h>
#include <photon/accel_struct.h>
#include <photon/device_context.h>

//  contexts
auto device = ns::Context::getInstance()->device(0);
auto devCtx = pt::DeviceContext::create(device);
auto alloc = device->defaultAllocator();
auto stream = ns::Stream(device);

//  TODO ...
```


## License
Photon is distributed under the terms of the [MIT License](LICENSE).
