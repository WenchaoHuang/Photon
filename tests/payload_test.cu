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
#pragma once

#include <photon/payload.cuh>

using RayDirType = pt::Payload<float3, 0, 1, 2>;

/*********************************************************************************
*********************************    kernels    **********************************
*********************************************************************************/

__RT_KERNEL__ void __raygen__()
{
	RayDirType rayDir;
	pt::set_payload<0>(0);
	pt::set_payload<long long, 0, 1>(0);
	pt::set_payload(rayDir);
}


__RT_KERNEL__ void __miss__()
{
	pt::set_payload<0>(0);
	auto p0 = pt::get_payload<0>();
	auto p1 = pt::get_payload<float, 0>();
	auto rayDir = pt::get_payload<RayDirType>();
	rayDir = float3{ 0, 1, 0 };
	pt::set_payload(rayDir);
}