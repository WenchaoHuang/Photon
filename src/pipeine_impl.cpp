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
#include "device_context.h"
#include <nucleus/logger.h>
#include <nucleus/stream.h>
#include <optix_stubs.h>

PHOTON_USING_NAMESPACE

namespace
{
	bool appendProgramToGroupDesc(const std::shared_ptr<ProgramImpl> & program, OptixProgramGroupDesc & programGroupDesc, Program::Type & groupType)
	{
		auto module = program->module();
		auto entryFunctionName = program->entryFunctionName().c_str();

		if (module == nullptr)
		{
			return false;
		}

		switch (program->type())
		{
			case Program::DirectCallable:
				if ((groupType == Program::HitGroup) || (programGroupDesc.callables.moduleDC != nullptr))
				{
					return false;
				}

				programGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
				programGroupDesc.callables.moduleDC = module->handle();
				programGroupDesc.callables.entryFunctionNameDC = entryFunctionName;
				groupType = Program::CallableGroup;

				return true;

			case Program::ContinuationCallable:
				if ((groupType == Program::HitGroup) || (programGroupDesc.callables.moduleCC != nullptr))
				{
					return false;
				}

				programGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
				programGroupDesc.callables.moduleCC = module->handle();
				programGroupDesc.callables.entryFunctionNameCC = entryFunctionName;
				groupType = Program::CallableGroup;

				return true;

			case Program::AnyHit:
				if ((groupType == Program::CallableGroup) || (programGroupDesc.hitgroup.moduleAH != nullptr))
				{
					return false;
				}

				programGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
				programGroupDesc.hitgroup.moduleAH = module->handle();
				programGroupDesc.hitgroup.entryFunctionNameAH = entryFunctionName;
				groupType = Program::HitGroup;

				return true;

			case Program::ClosestHit:
				if ((groupType == Program::CallableGroup) || (programGroupDesc.hitgroup.moduleCH != nullptr))
				{
					return false;
				}

				programGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
				programGroupDesc.hitgroup.moduleCH = module->handle();
				programGroupDesc.hitgroup.entryFunctionNameCH = entryFunctionName;
				groupType = Program::HitGroup;

				return true;

			case Program::Intersection:
				if ((groupType == Program::CallableGroup) || (programGroupDesc.hitgroup.moduleIS != nullptr))
				{
					return false;
				}

				programGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
				programGroupDesc.hitgroup.moduleIS = module->handle();
				programGroupDesc.hitgroup.entryFunctionNameIS = entryFunctionName;
				groupType = Program::HitGroup;

				return true;

			default:
				return false;
		}
	}
}

/*********************************************************************************
********************************    ModuleImpl    ********************************
*********************************************************************************/

ModuleImpl::ModuleImpl(std::shared_ptr<DeviceContext> deviceContext, OptixModule hModule) : m_deviceContext(deviceContext), m_hModule(hModule)
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

	auto program = std::make_shared<ProgramImpl>(this->shared_from_this(), hProgramGroup, progType, funcName);

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

ProgramImpl::ProgramImpl(std::shared_ptr<ModuleImpl> module, OptixProgramGroup hProgramGroup, Program::Type type, std::string funcName)
	: m_module(module), m_hProgramGroup(hProgramGroup), m_progType(type), m_funcName(std::move(funcName))
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

	if (!impl_0 || !impl_1)
	{
		NS_ERROR_LOG("Invalid program!");

		return nullptr;
	}

	auto deviceContext = impl_0->deviceContext();
	auto deviceContext_1 = impl_1->deviceContext();

	if ((deviceContext == nullptr) || (deviceContext_1 == nullptr) || (deviceContext->handle() != deviceContext_1->handle()))
	{
		NS_ERROR_LOG("Programs must belong to the same device context!");

		return nullptr;
	}

	OptixProgramGroup hProgramGroup = nullptr;
	OptixProgramGroupDesc programGroupDesc = {};
	OptixProgramGroupOptions programGroupOptions = {};
	Program::Type groupType = Program::Unknow;

	programGroupDesc.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;

	if (!appendProgramToGroupDesc(impl_0, programGroupDesc, groupType) || !appendProgramToGroupDesc(impl_1, programGroupDesc, groupType))
	{
		NS_ERROR_LOG("Programs cannot be combined!");

		return nullptr;
	}

	OptixResult err = optixProgramGroupCreate(deviceContext->handle(), &programGroupDesc, 1, &programGroupOptions, nullptr, nullptr, &hProgramGroup);

	if (err != OPTIX_SUCCESS)
	{
		NS_ERROR_LOG("%s.", optixGetErrorString(err));

		return nullptr;
	}

	return std::make_shared<ProgramImpl>(impl_0->module(), hProgramGroup, groupType);
}


std::shared_ptr<Program> Program::combine(std::shared_ptr<Program> program0, std::shared_ptr<Program> program1, std::shared_ptr<Program> program2)
{
	auto impl_0 = std::dynamic_pointer_cast<ProgramImpl>(program0);
	auto impl_1 = std::dynamic_pointer_cast<ProgramImpl>(program1);
	auto impl_2 = std::dynamic_pointer_cast<ProgramImpl>(program2);

	if (!impl_0 || !impl_1 || !impl_2)
	{
		NS_ERROR_LOG("Invalid program!");

		return nullptr;
	}

	auto deviceContext = impl_0->deviceContext();
	auto deviceContext_1 = impl_1->deviceContext();
	auto deviceContext_2 = impl_2->deviceContext();

	if ((deviceContext == nullptr) || (deviceContext_1 == nullptr) || (deviceContext_2 == nullptr)
		|| (deviceContext->handle() != deviceContext_1->handle()) || (deviceContext->handle() != deviceContext_2->handle()))
	{
		NS_ERROR_LOG("Programs must belong to the same device context!");

		return nullptr;
	}

	OptixProgramGroup hProgramGroup = nullptr;
	OptixProgramGroupDesc programGroupDesc = {};
	OptixProgramGroupOptions programGroupOptions = {};
	Program::Type groupType = Program::Unknow;

	programGroupDesc.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;

	if (!appendProgramToGroupDesc(impl_0, programGroupDesc, groupType) || !appendProgramToGroupDesc(impl_1, programGroupDesc, groupType) || !appendProgramToGroupDesc(impl_2, programGroupDesc, groupType))
	{
		NS_ERROR_LOG("Programs cannot be combined!");

		return nullptr;
	}

	OptixResult err = optixProgramGroupCreate(deviceContext->handle(), &programGroupDesc, 1, &programGroupOptions, nullptr, nullptr, &hProgramGroup);

	if (err != OPTIX_SUCCESS)
	{
		NS_ERROR_LOG("%s.", optixGetErrorString(err));

		return nullptr;
	}

	return std::make_shared<ProgramImpl>(impl_0->module(), hProgramGroup, groupType);
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


std::shared_ptr<DeviceContext> ProgramImpl::deviceContext() const
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
*********************************    Pipeline    *********************************
*********************************************************************************/

Pipeline::Pipeline(SharedContext context, ns::ArrayProxy<std::shared_ptr<Program>> programs,
				   const OptixPipelineCompileOptions & pipelineCompileOptions, const OptixPipelineLinkOptions & pipelineLinkOptions)
	: m_context(context), m_hPipeline(nullptr)
{
	std::vector<OptixProgramGroup> programGroups(programs.size());

	for (size_t i = 0; i < programGroups.size(); i++)
	{
		auto progImpl = std::dynamic_pointer_cast<ProgramImpl>(programs[i]);

		if (progImpl == nullptr)
		{
			NS_ERROR_LOG("Invalid program!");

			return;
		}
		else
		{
			programGroups[i] = progImpl->handle();
		}
	}

	OptixResult err = optixPipelineCreate(context->handle(), &pipelineCompileOptions, &pipelineLinkOptions, programGroups.data(), programs.size(), nullptr, nullptr, &m_hPipeline);

	if (err != OPTIX_SUCCESS)
	{
		NS_ERROR_LOG("%s.", optixGetErrorString(err));

		throw err;
	}
}


void Pipeline::doLaunch(ns::Stream & stream, const void * pipelineParams, size_t pipelineParamsSize, const OptixShaderBindingTable & sbt, unsigned int width, unsigned int height, unsigned int depth)
{
	OptixResult err = optixLaunch(m_hPipeline, stream.handle(), CUdeviceptr(pipelineParams), pipelineParamsSize, &sbt, width, height, depth);

	if (err != OPTIX_SUCCESS)
	{
		NS_ERROR_LOG("%s.", optixGetErrorString(err));

		throw err;
	}
}


Pipeline::~Pipeline()
{
	if (m_hPipeline != nullptr)
	{
		OptixResult err = optixPipelineDestroy(m_hPipeline);

		NS_ERROR_LOG_IF(err != OPTIX_SUCCESS, "%s.", optixGetErrorString(err));
	}
}
