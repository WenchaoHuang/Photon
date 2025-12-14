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

#include <nucleus/device.h>
#include <nucleus/context.h>
#include <nucleus/allocator.h>

#include <photon/denoiser.h>
#include <photon/device_context.h>

/*********************************************************************************
******************************    denoiser_test    *******************************
*********************************************************************************/

void denoiser_test()
{
	auto device = ns::Context::getInstance()->device(0);
	auto deviceContext = pt::SharedContext(device);
	auto denoiser = deviceContext->createDenoiser();
	auto allocator = device->defaultAllocator();

#if OPTIX_VERSION > 70500
	denoiser->preallocate(allocator, pt::Denoiser::TemporalUpscale2x, 1024, 1024);
	denoiser->preallocate(allocator, pt::Denoiser::Upscale2x, 1024, 1024);
#endif
#if OPTIX_VERSION > 70400
	denoiser->preallocate(allocator, pt::Denoiser::Temporal, 1024, 1024);
#endif
	denoiser->preallocate(allocator, pt::Denoiser::Normal, 1024, 1024);

	assert(denoiser->maxInputWidth() == 1024);
	assert(denoiser->maxInputHeight() == 1024);
}