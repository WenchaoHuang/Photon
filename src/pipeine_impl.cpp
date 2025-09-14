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
********************************    ModuleImpl    ********************************
*********************************************************************************/

ModuleImpl::ModuleImpl(std::shared_ptr<DeviceContextImpl> deviceContext, OptixModule hModule) : m_deviceContext(deviceContext), m_hModule(hModule)
{

}


std::shared_ptr<Program> ModuleImpl::at(const std::string & funcName)
{
	// 1. Validation
	auto progType = ProgramImpl::queryProgramType(funcName);

	if (funcName.empty())
	{
		NS_ERROR_LOG("Empty function name!");

		return nullptr;
	}
	else if ((progType == Program::Unknow) || (progType == Program::BuiltinIntersection))
	{
		NS_ERROR_LOG("Invalid function name: %s", funcName.c_str());

		return nullptr;
	}

	// 2. Check
	auto iter = m_programMap.find(funcName);

	if (iter != m_programMap.end())
	{
		if (!iter->second.expired())
		{
			auto program = iter->second.lock();

			if (program != nullptr)
			{
				return program;
			}
		}
	}

	// 3. Create single program.
	OptixProgramGroup hProgramGroup = nullptr;
	OptixProgramGroupDesc programGroupDesc = { .flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE };
	OptixProgramGroupOptions programGroupOptions = {};

	if (progType == Program::Raygen)
	{
		programGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		programGroupDesc.raygen.module = m_hModule;
		programGroupDesc.raygen.entryFunctionName = funcName.c_str();
	}
	else if (progType == Program::Miss)
	{
		programGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
		programGroupDesc.miss.module = m_hModule;
		programGroupDesc.miss.entryFunctionName = funcName.c_str();
	}
	else if (progType == Program::AnyHit)
	{
		programGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		programGroupDesc.hitgroup.moduleAH = m_hModule;
		programGroupDesc.hitgroup.entryFunctionNameAH = funcName.c_str();
	}
	else if (progType == Program::ClosestHit)
	{
		programGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		programGroupDesc.hitgroup.moduleCH = m_hModule;
		programGroupDesc.hitgroup.entryFunctionNameCH = funcName.c_str();
	}
	else if (progType == Program::Intersection)
	{
		programGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		programGroupDesc.hitgroup.moduleIS = m_hModule;
		programGroupDesc.hitgroup.entryFunctionNameIS = funcName.c_str();
	}
	else if (progType == Program::DirectCallable)
	{
		programGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
		programGroupDesc.callables.entryFunctionNameDC = funcName.c_str();
		programGroupDesc.callables.moduleDC = m_hModule;
	}
	else if (progType == Program::ContinuationCallable)
	{
		programGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
		programGroupDesc.callables.entryFunctionNameCC = funcName.c_str();
		programGroupDesc.callables.moduleCC = m_hModule;
	}
	else if (progType == Program::Exception)
	{
		programGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
		programGroupDesc.exception.module = m_hModule;
		programGroupDesc.exception.entryFunctionName = funcName.c_str();
	}

	OptixResult err = optixProgramGroupCreate(m_deviceContext->handle(), &programGroupDesc, 1, &programGroupOptions, nullptr, nullptr, &hProgramGroup);

	if (err != OPTIX_SUCCESS)
	{
		NS_ERROR_LOG_IF(err != OPTIX_SUCCESS, "%s.", optixGetErrorString(err));

		return nullptr;
	}

	auto program = std::make_shared<ProgramImpl>(this->shared_from_this(), hProgramGroup, progType);

	m_programMap[funcName] = program;

	return program;
}


ModuleImpl::~ModuleImpl()
{
	if (m_hModule != nullptr)
	{
		OptixResult err = optixModuleDestroy(m_hModule);

		NS_ERROR_LOG_IF(err != OPTIX_SUCCESS, "%s.", optixGetErrorString(err));
	}
}

/*********************************************************************************
*******************************    ProgramImpl    ********************************
*********************************************************************************/

ProgramImpl::ProgramImpl(std::shared_ptr<ModuleImpl> module, OptixProgramGroup hProgramGroup, Program::Type type)
	: m_module(module), m_hProgramGroup(hProgramGroup), m_progType(type)
{
	OptixResult err = optixSbtRecordPackHeader(m_hProgramGroup, m_header.storage);

	if (err != OPTIX_SUCCESS)
	{
		optixProgramGroupDestroy(m_hProgramGroup);

		NS_ERROR_LOG("%s.", optixGetErrorString(err));

		throw err;
	}
}


std::shared_ptr<Program> Program::combine(std::shared_ptr<Program> program0, std::shared_ptr<Program> program1)
{
	auto impl_0 = std::dynamic_pointer_cast<ProgramImpl>(program0);
	auto impl_1 = std::dynamic_pointer_cast<ProgramImpl>(program1);

	//	TODO

	return nullptr;
}


std::shared_ptr<Program> Program::combine(std::shared_ptr<Program> program0, std::shared_ptr<Program> program1, std::shared_ptr<Program> program2)
{
	auto impl_0 = std::dynamic_pointer_cast<ProgramImpl>(program0);
	auto impl_1 = std::dynamic_pointer_cast<ProgramImpl>(program1);
	auto impl_2 = std::dynamic_pointer_cast<ProgramImpl>(program2);

	if (!impl_0 || !impl_1 || !impl_2)
	{
		NS_ERROR_LOG("Invalid program!");
	}
	else
	{
		//	TODO
	}

	return nullptr;
}


Program::Type ProgramImpl::queryProgramType(const std::string & funcName)
{
	if (funcName.starts_with("__miss__"))							return Program::Miss;
	else if (funcName.starts_with("__raygen__"))					return Program::Raygen;
	else if (funcName.starts_with("__anyhit__"))					return Program::AnyHit;
	else if (funcName.starts_with("__exception__"))					return Program::Exception;
	else if (funcName.starts_with("__closesthit__"))				return Program::ClosestHit;
	else if (funcName.starts_with("__intersection__"))				return Program::Intersection;
	else if (funcName.starts_with("__direct_callable__"))			return Program::DirectCallable;
	else if (funcName.starts_with("__builtin_intersection__"))		return Program::BuiltinIntersection;
	else if (funcName.starts_with("__continuation_callable__"))		return Program::ContinuationCallable;
	else															return Program::Unknow;
}


std::shared_ptr<DeviceContextImpl> ProgramImpl::deviceContext() const
{
	return m_module ? m_module->deviceContext() : nullptr;
}


ProgramImpl::~ProgramImpl()
{
	if (m_hProgramGroup != nullptr)
	{
		OptixResult err = optixProgramGroupDestroy(m_hProgramGroup);

		NS_ERROR_LOG_IF(err != OPTIX_SUCCESS, "%s.", optixGetErrorString(err));
	}
}


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