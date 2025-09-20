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
#include <photon/macros.h>
#include <device_launch_parameters.h>
#include <optix_device.h>
#include "launch_params.h"

__RT_CONSTANT__ LaunchParams launchParams;

/*********************************************************************************
*********************************    kernels    **********************************
*********************************************************************************/

__RT_KERNEL__ void __raygen__()
{
	auto vIdx = optixGetLaunchIndex().x;
	auto p0 = launchParams.vertices[vIdx];

	optixTrace(launchParams.traversable,
				float3{ p0.x, p0.y, p0.z },
				float3{ 1.0f, 0.0f, 0.0f },
				0.0f,
				1e-6f,
				0.0f,
				OptixVisibilityMask(255),
				OPTIX_RAY_FLAG_DISABLE_ANYHIT,
				0,
				0,
				0);
}


__RT_KERNEL__ void __intersection__()
{
	auto p0 = optixGetWorldRayOrigin();
	auto vIdx0 = optixGetLaunchIndex().x;
	auto vIdx1 = optixGetPrimitiveIndex();

	if (vIdx1 <= vIdx0)
		return;

	auto p1 = launchParams.vertices[vIdx1];
	float3 d = { p1.x - p0.x, p1.y - p0.y, p1.z - p0.z };
	float len = sqrtf((d.x * d.x) + (d.y * d.y) + (d.z * d.z));

	if (len < launchParams.radius)
	{
		atomicAdd(launchParams.count.data(), 1);
	}
}


__RT_KERNEL__ void __miss__()
{
//	printf("miss\n");
}