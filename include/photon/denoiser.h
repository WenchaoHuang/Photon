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

#include "fwd.h"
#include <nucleus/vector_types.h>
#include <nucleus/device_pointer.h>

namespace PHOTON_NAMESPACE
{
	/*****************************************************************************
	*******************************    Denoiser    *******************************
	*****************************************************************************/

	/**
	 *	@brief		Abstract class for denoising implementations.
	 *	@note		Provides common interface for temporal and AOV-based denoising with optional 2x upscaling support.
	 */
	class Denoiser
	{

	public:

		//!	@brief		Default constructor.
		Denoiser() {}

		//!	@brief		Virtual destructor 
		virtual ~Denoiser() {}

	public:

		//	Model kind used by the denoiser.
		enum ModelKind
		{
			Normal				= 0,	//	Built-in model for denoising single image.
			Temporal			= 1,	//	Built-in model for denoising image sequence, temporally stable.
			Upscale2x			= 2,	//	Built-in model for denoising single image upscaling (supports AOVs).
			TemporalUpscale2x	= 3,	//	Built-in model for denoising image sequence upscaling, temporally stable (supports AOVs).
		};

		//!	@brief		Releases allocated resources.
		virtual void release() = 0;

		//!	@brief		Return model kind used by the denoiser.
		virtual ModelKind modelKind() const = 0;

		//!	@brief		Return maximum input width of the image.
		virtual unsigned int maxInputWidth() const = 0;

		//!	@brief		Return maximum input height of the image.
		virtual unsigned int maxInputHeight() const = 0;

		//!	@brief		Retrieve the device context associated with.
		virtual std::shared_ptr<class DeviceContext> deviceContext() = 0;

	public:

		/**
		 *	@brief		Pre-allocates GPU memory resources for denoising tasks.
		 *	@param[in]	pAlloc - Custom memory allocator.
		 *	@param[in]	modeKind - Model kind used by the desnoiser.
		 *	@param[in]	maxInputWidth - Maximum width of the input image in pixels.
		 *	@param[in]	maxInputHeight - Maximum height of the input image in pixels.
		 */
		virtual void preallocate(ns::AllocPtr pAlloc, ModelKind modeKind, unsigned int maxInputWidth, unsigned int maxInputHeight) = 0;


		/**
		 *	@brief		Execute temporal denoising pass.
		 *	@param[in]	stream - CUDA stream for asynchronous execution.
		 *	@param[out]	output - Denoised output image (RGBA32F).
		 *	@param[in]	input - Current noisy input image.
		 *	@param[in]	albedo - Current albedo buffer.
		 *	@param[in]	normal - Current normal buffer.
		 *	@param[in]	previousOutput - [optional] Previous frame's denoised result. 
		 *	@param[in]	flow - [optional] 2D motion vectors (XY32F). In the first frame, `flowImg` must be set to zero (no motion).
		 *	@param[in]	flowTrustworthiness - [optional] Motion vector confidence map (F32), Range 0..1 (low->high trustworthiness). Ignored if data pointer in the image is zero.
		 *	@param[in]	blendFactor - Field specifies an interpolation weight [0.0-1.0] between the noisy input image.
		 */
		virtual void launch(ns::Stream & stream, dev::Ptr2<Color4f> output, dev::Ptr2<const Color4f> input, dev::Ptr2<const Color4f> albedo, dev::Ptr2<const Color4f> normal,
							dev::Ptr2<const Color4f> previousOutput, dev::Ptr2<const ns::float2> flow, dev::Ptr2<const float> flowTrustworthiness, float blendFactor) = 0;


		/**
		 *	@brief		Advance to next temporal frame.
		 */
		virtual void nextFrame() = 0;
	};
}