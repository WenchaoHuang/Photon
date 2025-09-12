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
#include "device_context_impl.h"
#include <nucleus/logger.h>
#include <nucleus/stream.h>
#include <optix_stubs.h>

PHOTON_USING_NAMESPACE

/*********************************************************************************
*******************************    ProgramImpl    ********************************
*********************************************************************************/

PipelineImpl::PipelineImpl(std::shared_ptr<DeviceContextImpl> deviceContext, OptixPipeline hPipeline)
	: m_deviceContext(deviceContext), m_hPipeline(hPipeline)
{

}


void PipelineImpl::doLaunch(ns::Stream & stream, const void * pipelineParams, size_t pipelineParamsSize,
							const OptixShaderBindingTable & sbt, unsigned int width, unsigned int height, unsigned int depth)
{
	OptixResult err = optixLaunch(m_hPipeline, stream.handle(), CUdeviceptr(pipelineParams), pipelineParamsSize, &sbt, width, height, depth);

	if (err != OPTIX_SUCCESS)
	{
		NS_ERROR_LOG("%s.", optixGetErrorString(err));

		throw err;
	}
}


PipelineImpl::~PipelineImpl()
{
	if (m_hPipeline != nullptr)
	{
		OptixResult err = optixPipelineDestroy(m_hPipeline);

		NS_ERROR_LOG_IF(err != OPTIX_SUCCESS, "%s.", optixGetErrorString(err));
	}
}