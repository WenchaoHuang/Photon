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
#include <optix.h>
#include <nucleus/device_pointer.h>

namespace PHOTON_NAMESPACE
{
	/*****************************************************************************
	*******************************    Pipeline    *******************************
	*****************************************************************************/

	/**
	 *	@brief		Abstract interface for an OptiX pipeline.
	 *	@note		A pipeline represents the compiled set of program groups
	 *				(raygen, miss, hit, callable, etc.) and encapsulates the
	 *				execution configuration for launching OptiX kernels.
	 */
	class Pipeline
	{

	public:

		//!	Virtual destructor.
		virtual ~Pipeline() {}

		/**
		 *	@brief		Launch the pipeline with the given parameters.
		 *	@tparam		Type - Type of the pipeline parameter structure.
		 *	@param[in]	stream - CUDA stream to enqueue the launch on.
		 *	@param[in]	pipelineParams - Pointer to the pipeline parameter structure.
		 *	@param[in]	sbt - Shader Binding Table that defines program associations (raygen, miss, hit, callable, etc.).
		 *	@param[in]	width - Launch width in threads.
		 *	@param[in]	height - Launch height in threads.
		 *	@param[in]	depth - Launch depth in threads (default = 1).
		 *	@note		This function dispatches ray generation and subsequent OptiX programs across the specified launch dimensions.
		 *	@note		Multiple launches may be issued in parallel from multiple threads as long as they target different CUDA streams.
		 *	@warning	The stream and pipeline must belong to the same device context.
		 */
		template<typename Type> void launch(ns::Stream & stream, ns::dev::Ptr<const Type> pipelineParams, const OptixShaderBindingTable & sbt, size_t width, size_t height, size_t depth = 1)
		{
			this->doLaunch(stream, pipelineParams.data(), sizeof(Type), sbt, static_cast<unsigned int>(width), static_cast<unsigned int>(height), static_cast<unsigned int>(depth));
		}

	private:

		/**
		 *	@brief		Internal implementation of pipeline launch.
		 *	@note		This is the low-level entry point that forwards the call to the OptiX API with untyped pipeline parameters.
		 */
		virtual void doLaunch(ns::Stream & stream, const void * pipelineParams, size_t pipelineParamsSize,
							  const OptixShaderBindingTable & sbt, unsigned int width, unsigned int height, unsigned int depth) = 0;
	};
}