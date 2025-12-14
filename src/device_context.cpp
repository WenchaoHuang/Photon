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

#include "pipeline_impl.h"
#include "denoiser_impl.h"
#include "device_context.h"
#include "accel_struct_impl.h"

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

DeviceContext::DeviceContext(ns::Device * device, int logLevel, [[maybe_unused]] bool validationMode) : m_device(device), m_hContext(nullptr)
{
	OptixResult err = optixInit();

	if (err == OPTIX_SUCCESS)
	{
		device->init();

		OptixDeviceContextOptions						deviceContextOptions = {};
	#if OPTIX_VERSION >= 70200
		deviceContextOptions.validationMode				= validationMode ? OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL : OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
	#endif
		deviceContextOptions.logCallbackLevel			= NS_MIN(NS_MAX(0, logLevel), 4);	//!	0 -> Disable, 1 -> Fatal, 2 -> Error, 3 -> Warning, 4 -> Info.
		deviceContextOptions.logCallbackFunction		= optixLog;
		deviceContextOptions.logCallbackData			= device;

		err = optixDeviceContextCreate(nullptr, &deviceContextOptions, &m_hContext);

		if (err == OPTIX_SUCCESS)
		{
		#if (OPTIX_VERSION >= 70000)
			optixDeviceContextGetProperty(m_hContext, OPTIX_DEVICE_PROPERTY_RTCORE_VERSION, &m_devProp.version, sizeof(DeviceProp::version));
			optixDeviceContextGetProperty(m_hContext, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_OFFSET, &m_devProp.maxSbtOffset, sizeof(DeviceProp::maxSbtOffset));
			optixDeviceContextGetProperty(m_hContext, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH, &m_devProp.maxTraceDepth, sizeof(DeviceProp::maxTraceDepth));
			optixDeviceContextGetProperty(m_hContext, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID, &m_devProp.maxInstanceID, sizeof(DeviceProp::maxInstanceID));
			optixDeviceContextGetProperty(m_hContext, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS, &m_devProp.maxInstancesPerIAS, sizeof(DeviceProp::maxInstancesPerIAS));
			optixDeviceContextGetProperty(m_hContext, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS, &m_devProp.maxPrimitivesPerGAS, sizeof(DeviceProp::maxPrimitivesPerGAS));
			optixDeviceContextGetProperty(m_hContext, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS, &m_devProp.maxSbtRecordsPerGAS, sizeof(DeviceProp::maxSbtRecordsPerGAS));
			optixDeviceContextGetProperty(m_hContext, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH, &m_devProp.maxTraversableGraphDepth, sizeof(DeviceProp::maxTraversableGraphDepth));
			optixDeviceContextGetProperty(m_hContext, OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK, &m_devProp.numBitsInstanceVisiblityMask, sizeof(DeviceProp::numBitsInstanceVisiblityMask));
		#endif
		#if (OPTIX_VERSION >= 80100)
			optixDeviceContextGetProperty(m_hContext, OPTIX_DEVICE_PROPERTY_SHADER_EXECUTION_REORDERING, &m_devProp.shaderExecutionReordering, sizeof(DeviceProp::shaderExecutionReordering));
		#endif
		#if (OPTIX_VERSION >= 90000)
			optixDeviceContextGetProperty(m_hContext, OPTIX_DEVICE_PROPERTY_CLUSTER_ACCEL, &m_devProp.clusterAccel, sizeof(DeviceProp::clusterAccel));
			optixDeviceContextGetProperty(m_hContext, OPTIX_DEVICE_PROPERTY_COOP_VEC, &m_devProp.cooperativeVector, sizeof(DeviceProp::cooperativeVector));
			optixDeviceContextGetProperty(m_hContext, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_CLUSTER_VERTICES, &m_devProp.maxClusterVertices, sizeof(DeviceProp::maxClusterVertices));
			optixDeviceContextGetProperty(m_hContext, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_CLUSTER_TRIANGLES, &m_devProp.maxClusterTriangles, sizeof(DeviceProp::maxClusterTriangles));
			optixDeviceContextGetProperty(m_hContext, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_STRUCTURED_GRID_RESOLUTION, &m_devProp.maxStructuredGridResolution, sizeof(DeviceProp::maxStructuredGridResolution));
		#endif

			NS_INFO_LOG("Creating Optix context on device(%d) successfully, RT-Core version: %d.", device->id(), m_devProp.version);

			return;
		}
	}

	NS_ERROR_LOG("Failed to create Optix context on device(%d): %s.", device->id(), optixGetErrorString(err));

	throw err;
}


std::shared_ptr<Module> DeviceContext::createModule(const unsigned char * ptxStr, size_t ptxSize,
													const OptixModuleCompileOptions & moduleCompileOptions,
													const OptixPipelineCompileOptions & pipelineCompileOptions)
{
	OptixModule hModule = nullptr;

#if OPTIX_VERSION >= 70700
	OptixResult err = optixModuleCreate(m_hContext, &moduleCompileOptions, &pipelineCompileOptions, (const char*)ptxStr, ptxSize, nullptr, nullptr, &hModule);
#else
	OptixResult err = optixModuleCreateFromPTX(m_hContext, &moduleCompileOptions, &pipelineCompileOptions, (const char*)ptxStr, ptxSize, nullptr, nullptr, &hModule);
#endif

	if (err == OPTIX_SUCCESS)
	{
		return std::make_shared<ModuleImpl>(this->shared_from_this(), hModule);
	}

	NS_ERROR_LOG("%s.", optixGetErrorString(err));

	throw err;
}


std::unique_ptr<InstAccelStruct> DeviceContext::createInstAccelStruct()
{
	return std::make_unique<InstAccelStructImpl>(this->shared_from_this());
}


std::unique_ptr<AccelStructAabb> DeviceContext::createAccelStructAabb()
{
	return std::make_unique<AccelStructAabbImpl>(this->shared_from_this());
}


std::unique_ptr<AccelStructCurve> DeviceContext::createAccelStructCurve()
{
	return std::make_unique<AccelStructCurveImpl>(this->shared_from_this());
}


std::unique_ptr<AccelStructSphere> DeviceContext::createAccelStructSphere()
{
	return std::make_unique<AccelStructSphereImpl>(this->shared_from_this());
}


std::unique_ptr<AccelStructTriangle> DeviceContext::createAccelStructTriangle()
{
	return std::make_unique<AccelStructTriangleImpl>(this->shared_from_this());
}


std::unique_ptr<Denoiser> DeviceContext::createDenoiser()
{
	return std::make_unique<DenoiserImpl>(this->shared_from_this());
}


DeviceContext::~DeviceContext()
{
	if (m_hContext != nullptr)
	{
		OptixResult err = optixDeviceContextDestroy(m_hContext);

		NS_ERROR_LOG_IF(err != OPTIX_SUCCESS, "%s.", optixGetErrorString(err));
	}
}