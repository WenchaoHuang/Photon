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

#include "accel_struct.h"
#include "device_context.h"
#include <nucleus/utility.h>
#include <nucleus/stream.h>
#include <optix_stubs.h>

PHOTON_USING_NAMESPACE

/*********************************************************************************
*******************************    AccelStruct    ********************************
*********************************************************************************/

AccelStruct::AccelStruct(std::shared_ptr<DeviceContext> deviceContext) : m_deviceContext(deviceContext), m_hTraversable(0), m_headerSize(0)
{
	m_buildOptions = OptixAccelBuildOptions{};
}


void AccelStruct::build(ns::Stream & stream, ns::AllocPtr allocator, const std::vector<OptixBuildInput> & buildInputs, BuildOptions buildOptions)
{
	//	Compaction is intentionally not supported (see class doc for rationale).
	//	Strip the flag and warn if the caller passed it anyway.
	if (buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION)
	{
		buildOptions.buildFlags = OptixBuildFlags(buildOptions.buildFlags & ~OPTIX_BUILD_FLAG_ALLOW_COMPACTION);

		NS_WARNING_LOG("OPTIX_BUILD_FLAG_ALLOW_COMPACTION is not supported and will be ignored.");
	}

	OptixAccelBufferSizes accelBufferSizes = {};
	OptixAccelBuildOptions optixBuildOptions = {};

	optixBuildOptions.buildFlags = buildOptions.buildFlags;
	optixBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
	optixBuildOptions.motionOptions = buildOptions.motionOptions;

	OptixResult err = optixAccelComputeMemoryUsage(m_deviceContext->handle(), &optixBuildOptions, buildInputs.data(), (uint32_t)buildInputs.size(), &accelBufferSizes);

	if (err == OPTIX_SUCCESS)
	{
		buildOptions.headerSize = ns::align_up(buildOptions.headerSize, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT);

		m_tempBuffer.resize(allocator, NS_MAX(accelBufferSizes.tempSizeInBytes, accelBufferSizes.tempUpdateSizeInBytes));

		//!	First \p headerSize bytes for storing user data.
		m_outputBuffer.resize(allocator, buildOptions.headerSize + accelBufferSizes.outputSizeInBytes);

		err = optixAccelBuild(m_deviceContext->handle(), stream.handle(), &optixBuildOptions, buildInputs.data(), (uint32_t)buildInputs.size(),
							  (CUdeviceptr)m_tempBuffer.data(), m_tempBuffer.size_bytes(), CUdeviceptr(m_outputBuffer.data() + buildOptions.headerSize),
							  m_outputBuffer.size_bytes() - buildOptions.headerSize, &m_hTraversable, nullptr, 0);
	}

	if (err != OPTIX_SUCCESS)
	{
		NS_ERROR_LOG("Failed to build acceleration structure: %s.", optixGetErrorString(err));

		throw err;
	}

	m_headerSize = buildOptions.headerSize;
	m_buildOptions = optixBuildOptions;
}


void AccelStruct::rebuild(ns::Stream & stream, const std::vector<OptixBuildInput> & buildInputs)
{
	if (m_hTraversable != 0)
	{
		OptixResult err = OPTIX_SUCCESS;

		m_buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

		err = optixAccelBuild(m_deviceContext->handle(), stream.handle(), &m_buildOptions, buildInputs.data(), (uint32_t)buildInputs.size(),
							  (CUdeviceptr)m_tempBuffer.data(), m_tempBuffer.size_bytes(), CUdeviceptr(m_outputBuffer.data() + m_headerSize),
							  m_outputBuffer.size_bytes() - m_headerSize, &m_hTraversable, nullptr, 0);

		if (err != OPTIX_SUCCESS)
		{
			NS_ERROR_LOG("Failed to rebuild acceleration structure: %s.", optixGetErrorString(err));

			throw err;
		}
	}
}


void AccelStruct::refit(ns::Stream & stream, const std::vector<OptixBuildInput> & buildInputs)
{
	OptixResult err = OPTIX_SUCCESS;

	if (this->allowUpdate() && (m_hTraversable != 0))
	{
		m_buildOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;

		err = optixAccelBuild(m_deviceContext->handle(), stream.handle(), &m_buildOptions, buildInputs.data(), (uint32_t)buildInputs.size(),
							  (CUdeviceptr)m_tempBuffer.data(), m_tempBuffer.size_bytes(), CUdeviceptr(m_outputBuffer.data() + m_headerSize),
							  m_outputBuffer.size_bytes() - m_headerSize, &m_hTraversable, nullptr, 0);

		if (err != OPTIX_SUCCESS)
		{
			NS_ERROR_LOG("Failed to refit acceleration structure: %s.", optixGetErrorString(err));

			throw err;
		}
	}
	else if (!this->allowUpdate())
	{
		NS_WARNING_LOG("Acceleration structure is non-updatable!");
	}
}