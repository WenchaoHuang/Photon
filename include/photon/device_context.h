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

namespace PHOTON_NAMESPACE
{
	/*****************************************************************************
	******************************    DeviceProp    ******************************
	*****************************************************************************/

	struct DeviceProp
	{
		//!	Optix 7.0.0
		unsigned int	version;							//!	The RT core version supported by the device (0 for no support, 10 for version 1.0).
		unsigned int	clusterAccel;						//!	Flag specifying support for cluster acceleration structure builds.
		unsigned int	maxSbtOffset;						//!	The maximum value for OptixInstance::sbtOffset.
		unsigned int	maxInstanceID;						//!	The maximum value for OptixInstance::instanceId.
		unsigned int	maxTraceDepth;						//!	Maximum value for OptixPipelineLinkOptions::maxTraceDepth.
		unsigned int	maxPrimitivesPerGAS;				//!	The maximum number of primitives (over all build inputs) as input to a single Geometry Acceleration Structure (GAS).
		unsigned int	maxInstancesPerIAS;					//!	The maximum number for the sum of the number of SBT records of all build inputs to a single Geometry Acceleration Structure (GAS).
		unsigned int	maxSbtRecordsPerGAS;				//!	The maximum number of instances that can be added to a single Instance Acceleration Structure (IAS).
		unsigned int	maxTraversableGraphDepth;			//!	Maximum value to pass into optixPipelineSetStackSize.
		unsigned int	numBitsInstanceVisiblityMask;		//!	The number of bits available for the OptixInstance::visibilityMask.
		//!	Optix 8.1.0
		unsigned int	shaderExecutionReordering;			//!	Flag specifying capabilities of the optixReorder() device function.
		//!	Optix 9.0.0
		unsigned int	cooperativeVector;					//!	Flag specifying whether cooperative vector support is enabled for this device.
		unsigned int	maxClusterVertices;					//!	The maximum unique vertices per cluster in a cluster acceleration structure builds.
		unsigned int	maxClusterTriangles;				//!	The maximum triangles per cluster in a cluster acceleration structure builds.
		unsigned int	maxStructuredGridResolution;		//!	The maximum resolution per cluster in a structured cluster acceleration structure builds.
	};

	/*****************************************************************************
	****************************    DeviceContext    *****************************
	*****************************************************************************/

	/**
	 *	@brief		RAII wrapper for OptiX device context.
	 * 
	 *	This class represents a wrapper around an OptiX device context.
	 *	It provides access to device-specific properties and the underlying device itself.
	 *	Derived implementations are responsible for managing the actual OptiX context and
	 *	associated resources such as modules, pipelines, and acceleration structures.
	 */
	class DeviceContext : public std::enable_shared_from_this<DeviceContext>
	{
		NS_NONCOPYABLE(DeviceContext)

	public:

		/**
		 *	@brief		Create a context for the device.
		 *	@param[in]	device - Pointer to the device associated with.
		 *	@param[in]	logLevel - Logging level for OptiX messages:
		 *				 0 - disable	Setting the callback level will disable all messages.  The callback
		 *								function will not be called in this case.
		 *				 1 - fatal		A non-recoverable error. The context and/or OptiX itself might no longer
		 *								be in a usable state.
		 *				 2 - error		A recoverable error, e.g., when passing invalid call parameters.
		 *				 3 - warning	Hints that OptiX might not behave exactly as requested by the user or
		 *								may perform slower than expected.
		 *				 4 - print		Status or progress messages.
		 *	@param[in]	validationMode - Enable or disable validation mode (requires Optix version >= 7.2.0).
		 *	@return		Return shared pointer to the newly created context.
		 *	@throw		OptixResult - Throw `OptixResult` in case of failure.
		 */
		PHOTON_API explicit DeviceContext(ns::Device * device, int logLevel = 3, bool validationMode = false);

		//!	@brief	Destructor.
		PHOTON_API ~DeviceContext();

	public:

		//!	@brief		Return pointer to the device associated with.
		ns::Device * device() const { return m_device; }

		//!	@brief		Return the native handle.
		OptixDeviceContext handle() { return m_hContext; }

		//!	@brief		Return pointer to the properties.
		const DeviceProp & properties() const { return m_devProp; }

		//!	@brief		Create a module from a PTX string.
		PHOTON_API std::shared_ptr<Module> createModule(const unsigned char * ptxStr, size_t ptxSize,
														const OptixModuleCompileOptions & moduleCompileOptions = OptixModuleCompileOptions{},
														const OptixPipelineCompileOptions & pipelineCompileOptions = OptixPipelineCompileOptions{});
		
		//! @brief  Create a module from a PTX string (array overload).
		//! This overload automatically infers the PTX size from the array, making it convenient to use PTX data generated by tools such as `bin2c.exe`.
		template<size_t ptxSize> std::shared_ptr<Module> createModule(const unsigned char(&ptx)[ptxSize],
																	  const OptixModuleCompileOptions & moduleCompileOptions = OptixModuleCompileOptions{},
																	  const OptixPipelineCompileOptions & pipelineCompileOptions = OptixPipelineCompileOptions{})
		{
			return this->createModule(ptx, ptxSize, moduleCompileOptions, pipelineCompileOptions);
		}

		//!	@brief		Create accel structs.
		PHOTON_API std::unique_ptr<InstAccelStruct> createInstAccelStruct();
		PHOTON_API std::unique_ptr<AccelStructAabb> createAccelStructAabb();
		PHOTON_API std::unique_ptr<AccelStructTriangle> createAccelStructTriangle();
	#if OPTIX_VERSION >= 70100
		PHOTON_API std::unique_ptr<AccelStructCurve> createAccelStructCurve();
	#endif
	#if OPTIX_VERSION >= 70500
		PHOTON_API std::unique_ptr<AccelStructSphere> createAccelStructSphere();
	#endif

		//! @brief		Create a denoiser.
		PHOTON_API std::unique_ptr<Denoiser> createDenoiser();

	private:

		ns::Device * const			m_device;
		OptixDeviceContext			m_hContext;
		DeviceProp					m_devProp;
	};
}