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
#include "sbt_record.h"
#include <nucleus/device_pointer.h>
#include <optix.h>
#include <string>

namespace PHOTON_NAMESPACE
{
	/*****************************************************************************
	********************************    Module    ********************************
	*****************************************************************************/
	
	/**
	 *	@brief		Abstract interface for an OptiX module.
	 *	@note		Represents a compiled OptiX module that contains one or more
	 *				program entry points. Programs can be retrieved by name.
	 */
	class Module
	{

	public:

		//!	Virtual destructor.
		virtual ~Module() {}


		/**
		 *	@brief		Retrieve a program by its function entry name.
		 *	@note		The function name must match one of the PTX entry points defined
		 *				in this module, such as "__raygen__xxx" or "__miss__yyy".
		 * @param[in]	funcName - The PTX function entry name.
		 * @return		A shared pointer to the corresponding Program.
		 */
		virtual std::shared_ptr<Program> at(const std::string & funcName) = 0;
	};

	/*****************************************************************************
	*******************************    Program    ********************************
	*****************************************************************************/
	
	/**
	 *	@brief		Abstract interface for an OptiX program.
	 *	@note		Represents a single OptiX program entry, such as a ray-generation,
	 *				miss, hit, or callable program. Each program has a type and an
	 *				associated SBT (Shader Binding Table) header.
	 */
	class Program
	{

	public:

		/**
		 *	@brief	Supported OptiX program types.
		 *	@note	Each type corresponds to a specific OptiX program entry function prefix or grouping.
		 */
		enum Type
		{
			Miss,						//	prefix "__miss__"
			AnyHit,						//	prefix "__anyhit__"
			Raygen,						//	prefix "__raygen__"
			Exception,					//	prefix "__exception__"
			ClosestHit,					//	prefix "__closesthit__"
			Intersection,				//	prefix "__intersection__"
			DirectCallable,				//	prefix "__direct_callable__"
			ContinuationCallable,		//	prefix "__continuation_callable__"
			BuiltinIntersection,
			CallableGroup,
			HitGroup,
			Unknow,
		};

		//!	Virtual destructor.
		virtual ~Program() {}

		//!	Get the type of this program.
		virtual Type type() const = 0;

		//!	Get the SBT header for this program.
		virtual const SbtHeader & header() const = 0;

		//	TODO
		PHOTON_API static std::shared_ptr<Program> combine(std::shared_ptr<Program> program0, std::shared_ptr<Program> program1);
		PHOTON_API static std::shared_ptr<Program> combine(std::shared_ptr<Program> program0, std::shared_ptr<Program> program1, std::shared_ptr<Program> program2);
	};

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