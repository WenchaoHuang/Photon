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

#include "module_impl.h"
#include "denoiser_impl.h"
#include "device_context_impl.h"

#include <nucleus/device.h>
#include <nucleus/logger.h>

#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#if OPTIX_VERSION < 70000
	#error "Requires Optix version >= 7.0.0!"
#endif

PHOTON_USING_NAMESPACE

/*********************************************************************************
*********************************    optixLog    *********************************
*********************************************************************************/

static void optixLog(unsigned int level, const char * tag, const char * msg, [[maybe_unused]] void * userData)
{
	switch (level)
	{
		case 1:		NS_ASSERT_LOG("[%s]: %s", tag, msg);		NS_ASSERT(false);		break;
		case 2:		NS_ERROR_LOG("[%s]: %s", tag, msg);									break;
		case 3:		NS_WARNING_LOG("[%s]: %s", tag, msg);								break;
		case 4:		NS_INFO_LOG("[%s]: %s", tag, msg);									break;
		default:																		break;
	}
};

/*********************************************************************************
******************************    DeviceContext    *******************************
*********************************************************************************/

std::shared_ptr<DeviceContext> DeviceContext::create(ns::Device * device, int logLevel, [[maybe_unused]] bool validationMode)
{
	OptixResult err = optixInit();

	if (err == OPTIX_SUCCESS)
	{
		device->init();

		OptixDeviceContext								hContext = nullptr;
		OptixDeviceContextOptions						deviceContextOptions = {};
	#if OPTIX_VERSION >= 70200
		deviceContextOptions.validationMode				= validationMode ? OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL : OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
	#endif
		deviceContextOptions.logCallbackLevel			= NS_MIN(NS_MAX(0, logLevel), 4);	//!	0 -> Disable, 1 -> Fatal, 2 -> Error, 3 -> Warning, 4 -> Info.
		deviceContextOptions.logCallbackFunction		= optixLog;
		deviceContextOptions.logCallbackData			= device;

		err = optixDeviceContextCreate(nullptr, &deviceContextOptions, &hContext);

		if (err == OPTIX_SUCCESS)
		{
			DeviceProp devProp = {};

		#if (OPTIX_VERSION >= 70000)
			optixDeviceContextGetProperty(hContext, OPTIX_DEVICE_PROPERTY_RTCORE_VERSION, &devProp.version, sizeof(DeviceProp::version));
			optixDeviceContextGetProperty(hContext, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_OFFSET, &devProp.maxSbtOffset, sizeof(DeviceProp::maxSbtOffset));
			optixDeviceContextGetProperty(hContext, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH, &devProp.maxTraceDepth, sizeof(DeviceProp::maxTraceDepth));
			optixDeviceContextGetProperty(hContext, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID, &devProp.maxInstanceID, sizeof(DeviceProp::maxInstanceID));
			optixDeviceContextGetProperty(hContext, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS, &devProp.maxInstancesPerIAS, sizeof(DeviceProp::maxInstancesPerIAS));
			optixDeviceContextGetProperty(hContext, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS, &devProp.maxPrimitivesPerGAS, sizeof(DeviceProp::maxPrimitivesPerGAS));
			optixDeviceContextGetProperty(hContext, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS, &devProp.maxSbtRecordsPerGAS, sizeof(DeviceProp::maxSbtRecordsPerGAS));
			optixDeviceContextGetProperty(hContext, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH, &devProp.maxTraversableGraphDepth, sizeof(DeviceProp::maxTraversableGraphDepth));
			optixDeviceContextGetProperty(hContext, OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK, &devProp.numBitsInstanceVisiblityMask, sizeof(DeviceProp::numBitsInstanceVisiblityMask));
		#endif
		#if (OPTIX_VERSION >= 80100)
			optixDeviceContextGetProperty(hContext, OPTIX_DEVICE_PROPERTY_SHADER_EXECUTION_REORDERING, &devProp.shaderExecutionReordering, sizeof(DeviceProp::shaderExecutionReordering));
		#endif
		#if (OPTIX_VERSION >= 90000)
			optixDeviceContextGetProperty(hContext, OPTIX_DEVICE_PROPERTY_CLUSTER_ACCEL, &devProp.clusterAccel, sizeof(DeviceProp::clusterAccel));
			optixDeviceContextGetProperty(hContext, OPTIX_DEVICE_PROPERTY_COOP_VEC, &devProp.cooperativeVector, sizeof(DeviceProp::cooperativeVector));
			optixDeviceContextGetProperty(hContext, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_CLUSTER_VERTICES, &devProp.maxClusterVertices, sizeof(DeviceProp::maxClusterVertices));
			optixDeviceContextGetProperty(hContext, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_CLUSTER_TRIANGLES, &devProp.maxClusterTriangles, sizeof(DeviceProp::maxClusterTriangles));
			optixDeviceContextGetProperty(hContext, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_STRUCTURED_GRID_RESOLUTION, &devProp.maxStructuredGridResolution, sizeof(DeviceProp::maxStructuredGridResolution));
		#endif

			NS_INFO_LOG("Creating Optix context on device(%d) successfully, RT-Core version: %d.", device->id(), devProp.version);

			return std::make_shared<DeviceContextImpl>(device, hContext, devProp);
		}
	}

	NS_ERROR_LOG("Failed to create Optix context on device(%d): %s.", device->id(), optixGetErrorString(err));

	throw err;
}


std::shared_ptr<Module> DeviceContextImpl::createModule(const unsigned char * ptxStr, size_t ptxSize,
														const OptixModuleCompileOptions & moduleCompileOptions,
														const OptixPipelineCompileOptions & pipelineCompileOptions)
{
	OptixModule hModule = nullptr;

	OptixResult eResult = optixModuleCreate(m_hContext, &moduleCompileOptions, &pipelineCompileOptions, (const char*)ptxStr, ptxSize, nullptr, nullptr, &hModule);

	if (eResult == OPTIX_SUCCESS)
	{
		return std::make_shared<ModuleImpl>(this->shared_from_this(), hModule);
	}

	NS_ERROR_LOG("Failed to create Optix module: %s.", optixGetErrorString(eResult));

	throw eResult;
}


std::unique_ptr<Denoiser> DeviceContextImpl::createDenoiser()
{
	return std::make_unique<DenoiserImpl>(this->shared_from_this());
}

/*********************************************************************************
****************************    DeviceContextImpl    *****************************
*********************************************************************************/

DeviceContextImpl::DeviceContextImpl(ns::Device * device, OptixDeviceContext hContext, const DeviceProp & devProp)
	: m_device(device), m_hContext(hContext), m_devProp(devProp)
{

}


DeviceContextImpl::~DeviceContextImpl()
{
	if (m_hContext != nullptr)
	{
		OptixResult err = optixDeviceContextDestroy(m_hContext);

		NS_ERROR_LOG_IF(err != OPTIX_SUCCESS, "%s.", optixGetErrorString(err));
	}
}