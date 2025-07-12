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
#include <nucleus/logger.h>
#include "internal/context_impl.h"

#include <optix_stubs.h>
#include <optix_function_table_definition.h>

using namespace photon;

/*********************************************************************************
*********************************    Context    **********************************
*********************************************************************************/

std::shared_ptr<Context> Context::create(ns::Device * device)
{
	const auto pfnLogCallback = [](unsigned int level, [[maybe_unused]] const char * tag, const char * msg, [[maybe_unused]] void * userData)
	{
		NS_ERROR_LOG_IF(level == 2, msg);		NS_WARNING_LOG_IF(level == 3, msg);		NS_INFO_LOG_IF(level == 4, msg);
	};

	OptixResult eResult = optixInit();

	if (eResult == OPTIX_SUCCESS)
	{
		device->init();

		OptixDeviceContext								hContext = nullptr;
		OptixDeviceContextOptions						deviceContextOptions = {};
		deviceContextOptions.validationMode				= OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
	//	deviceContextOptions.validationMode				= OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
		deviceContextOptions.logCallbackFunction		= pfnLogCallback;
		deviceContextOptions.logCallbackData			= device;
		deviceContextOptions.logCallbackLevel			= 3;	//!	0 -> Disable, 1 -> Fatal, 2 -> Error, 3 -> Warning, 4 -> Info.

		eResult = optixDeviceContextCreate(nullptr, &deviceContextOptions, &hContext);

		if (eResult == OPTIX_SUCCESS)
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

			NS_INFO_LOG("Creating Optix context on device(%d) successfully, RT-Core version: %d.", device->getID(), devProp.version);

			return std::make_shared<ContextImpl>(device, hContext, devProp);
		}
	}

	NS_ERROR_LOG("Failed to create Optix context on device(%d): %s.", device->getID(), optixGetErrorString(eResult));

	throw eResult;
}

/*********************************************************************************
*******************************    ContextImpl    ********************************
*********************************************************************************/

ContextImpl::ContextImpl(ns::Device * device, OptixDeviceContext hContext, const DeviceProp & devProp)
	: m_device(device), m_hContext(hContext), m_devProp(devProp)
{

}


ContextImpl::~ContextImpl()
{

}