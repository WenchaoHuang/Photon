/**
 *	Copyright (c) 2025 Wenchao Huang <physhuangwenchao@gmail.com>
 *
 *	Permission is hereby granted, free of charge, to any person obtaining a copy
 *	of this software and associated documentation files (the "Software"), to deal
 *	in the Software without restriction, including without limitation the rights
 *	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *	copies of the Software, and to permit persons to whom the Software is
 *	furnished to do so, subject to the following conditions:
 *
 *	The above copyright notice and this permission notice shall be included in all
 *	copies or substantial portions of the Software.
 *
 *	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *	SOFTWARE.
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../deps/optix-dev/include/optix_device.h"
#include "launch_params.h"

static __constant__ LaunchParams launchParams;

extern "C"
{
	__global__ void __raygen__()
	{
		int seed = launchParams.seed;

		printf("seed[%d] = %d\n", optixGetLaunchIndex().x, seed);
	}

	__global__ void __anyhit__()
	{

	}

	__global__ void __closesthit__()
	{

	}

	__global__ void __intersection__()
	{

	}

	__global__ void __direct_callable__()
	{

	}

	__global__ void __continuation_callable__()
	{

	}

	__global__ void __exception__()
	{

	}

	__global__ void __miss__()
	{

	}
}