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
#include <nucleus/span.h>
#include <nucleus/device_pointer.h>
#include <optix.h>
#include <string>

namespace PHOTON_NAMESPACE
{
	//!	Lightweight reference to a shader entry point.
	//! Does not create any OptiX resources. Pass to Program factory methods to create program groups.
	struct ProgramEntry
	{
		std::shared_ptr<Module>		module;		// The module that contains this entry point
		std::string					name;		// The function name of the entry point (e.g. "__raygen__", "__closesthit__xxx")
	};

	/*****************************************************************************
	********************************    Module    ********************************
	*****************************************************************************/

	/**
	 *	@brief		Abstract interface for an OptiX module.
	 *	@note		Represents a compiled OptiX module that contains one or more
	 *				program entry points.
	 */
	class Module
	{

	public:

		//!	Virtual destructor.
		virtual ~Module() {}

		/**
		 *	@brief		Get a lightweight entry reference by function name.
		 *	@note		Does not create any OptiX resources. The returned ProgramEntry should
		 *				be passed to Program factory methods (raygen, miss, hitgroup, etc.).
		 *	@param[in]	funcName - The PTX function entry name (e.g. "__raygen__", "__closesthit__xxx").
		 *	@return		A valid ProgramEntry on success, or an invalid (empty) ProgramEntry on failure.
		 */
		virtual ProgramEntry entry(const std::string & funcName) = 0;
	};

	/*****************************************************************************
	*******************************    Program    ********************************
	*****************************************************************************/

	/**
	 *	@brief		Represents an OptiX program group.
	 *	@note		Each Program owns an OptixProgramGroup and its SBT header.
	 *				Create via static factory methods: raygen(), miss(), exception(), hitgroup(), callables().
	 */
	class Program
	{

	public:

		//!	Virtual destructor.
		virtual ~Program() {}

		//!	Get the SBT header for this program.
		virtual const SbtHeader & header() const = 0;

		//!	@brief		Create a miss program group.
		PHOTON_API static std::shared_ptr<Program> miss(const ProgramEntry & ms);

		//!	@brief		Create a raygen program group.
		PHOTON_API static std::shared_ptr<Program> raygen(const ProgramEntry & rg);

		//!	@brief		Create an exception program group.
		PHOTON_API static std::shared_ptr<Program> exception(const ProgramEntry & ex);

		//!	@brief		Create a callables program group. Either entry can be empty.
		PHOTON_API static std::shared_ptr<Program> callables(const ProgramEntry & dc, const ProgramEntry & cc);

		//!	@brief		Create a hitgroup program group. Any entry can be empty (default ProgramEntry{}).
		PHOTON_API static std::shared_ptr<Program> hitgroup(const ProgramEntry & is, const ProgramEntry & ah, const ProgramEntry & ch);
	};

	/*****************************************************************************
	*******************************    Pipeline    *******************************
	*****************************************************************************/

	/**
	 *	@brief		Wrapper for an OptiX pipeline.
	 *	@note		A pipeline represents the compiled set of program groups
	 *				(raygen, miss, hit, callable, etc.) and encapsulates the
	 *				execution configuration for launching OptiX kernels.
	 */
	class Pipeline
	{
		NS_NONCOPYABLE(Pipeline)

	public:

		//!	@brief	Default constructor creates an invalid pipeline.
		Pipeline() : m_context(nullptr), m_hPipeline(nullptr) {}

		//!	@brief	
		PHOTON_API explicit Pipeline(SharedContext context,
									 ns::Span<const std::shared_ptr<Program>> programs,
									 const OptixPipelineCompileOptions & pipelineCompileOptions = OptixPipelineCompileOptions{},
									 const OptixPipelineLinkOptions & pipelineLinkOptions = OptixPipelineLinkOptions{});

		//!	@brief	Move constructor.
		Pipeline(Pipeline && rhs) noexcept : m_context(std::exchange(rhs.m_context, nullptr)), m_hPipeline(std::exchange(rhs.m_hPipeline, nullptr)) {}

		//!	@brief	Move assignment operator.
		void operator=(Pipeline && rhs) noexcept { m_context = std::exchange(rhs.m_context, nullptr);	m_hPipeline = std::exchange(rhs.m_hPipeline, nullptr); }

		//!	@brief	Destructor.
		PHOTON_API ~Pipeline();

	public:

		/**
		 *	@brief		Check if the pipeline is valid (i.e., successfully created).
		 */
		bool isValid() const { return m_hPipeline != nullptr; }


		/**
		 *	@brief		Explicit bool operator to check pipeline validity.
		 *	@returns	`true` if the pipeline is valid (i.e., has a non-null handle), `false` otherwise.
		 *	@note		Allows usage like: `if (pipeline) { ... }`
		 */
		operator bool() const { return m_hPipeline != nullptr; }


		/**
		 *	@brief		Launch the pipeline with the given parameters.
		 *	@tparam		Type - Type of the pipeline parameter structure.
		 *	@param[in]	stream - CUDA stream to enqueue the launch on.
		 *	@param[in]	pipelineParams - Pointer to the pipeline parameter structure.
		 *	@param[in]	sbt - Shader Binding Table that defines program associations (raygen, miss, hit, callable, etc.).
		 *	@param[in]	width - Launch width in threads.
		 *	@param[in]	height - Launch height in threads (default = 1).
		 *	@param[in]	depth - Launch depth in threads (default = 1).
		 *	@retval		ns::Stream& - Reference to this stream (enables method chaining).
		 *	@note		This function dispatches ray generation and subsequent OptiX programs across the specified launch dimensions.
		 *	@note		Multiple launches may be issued in parallel from multiple threads as long as they target different CUDA streams.
		 *	@warning	The stream and pipeline must belong to the same device context.
		 */
		template<typename Type> ns::Stream & launch(ns::Stream & stream, ns::dev::Ptr<const Type> pipelineParams, const OptixShaderBindingTable & sbt, unsigned int width, unsigned int height = 1, unsigned int depth = 1)
		{
			this->doLaunch(stream, pipelineParams.data(), sizeof(Type), sbt, width, height, depth);

			return stream;
		}

	private:

		/**
		 *	@brief		Internal implementation of pipeline launch.
		 *	@note		This is the low-level entry point that forwards the call to the OptiX API with untyped pipeline parameters.
		 */
		PHOTON_API void doLaunch(ns::Stream & stream, const void * pipelineParams, size_t pipelineParamsSize, const OptixShaderBindingTable & sbt, unsigned int width, unsigned int height, unsigned int depth);

	private:

		SharedContext		m_context;
		OptixPipeline		m_hPipeline;
	};
}