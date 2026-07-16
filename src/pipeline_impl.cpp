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

/*********************************************************************************
*****************************    queryProgramType    *****************************
*********************************************************************************/

static ProgramType queryProgramType(const std::string & funcName)
{
	if (funcName.starts_with("__miss__"))							return ProgramType::Miss;
	else if (funcName.starts_with("__raygen__"))					return ProgramType::Raygen;
	else if (funcName.starts_with("__anyhit__"))					return ProgramType::AnyHit;
	else if (funcName.starts_with("__exception__"))					return ProgramType::Exception;
	else if (funcName.starts_with("__closesthit__"))				return ProgramType::ClosestHit;
	else if (funcName.starts_with("__intersection__"))				return ProgramType::Intersection;
	else if (funcName.starts_with("__direct_callable__"))			return ProgramType::DirectCallable;
	else if (funcName.starts_with("__builtin_intersection__"))		return ProgramType::BuiltinIntersection;
	else if (funcName.starts_with("__continuation_callable__"))		return ProgramType::ContinuationCallable;
	else															return ProgramType::Unknow;
}

/*********************************************************************************
********************************    ModuleImpl    ********************************
*********************************************************************************/

ModuleImpl::ModuleImpl(std::shared_ptr<DeviceContext> deviceContext, OptixModule hModule) : m_deviceContext(deviceContext), m_hModule(hModule)
{

}


ProgramEntry ModuleImpl::entry(const std::string & funcName)
{
	auto progType = queryProgramType(funcName);

	if (funcName.empty())
	{
		NS_ERROR_LOG("Empty function name!");

		return ProgramEntry(nullptr, "");
	}
	else if ((progType == ProgramType::Unknow) || (progType == ProgramType::BuiltinIntersection))
	{
		NS_ERROR_LOG("Invalid function name: %s", funcName.c_str());

		return ProgramEntry(nullptr, "");
	}

	return ProgramEntry(this->shared_from_this(), funcName);
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

ProgramImpl::ProgramImpl(std::shared_ptr<DeviceContext> context, OptixProgramGroup hProgramGroup, ProgramType type)
	: m_deviceContext(context), m_hProgramGroup(hProgramGroup), m_progType(type)
{
	OptixResult err = optixSbtRecordPackHeader(m_hProgramGroup, m_header.storage);

	if (err != OPTIX_SUCCESS)
	{
		NS_ERROR_LOG("%s.", optixGetErrorString(err));

		throw err;
	}
}


std::shared_ptr<Program> Program::raygen(const ProgramEntry & entry)
{
	if (!entry.module || (queryProgramType(entry.name) != ProgramType::Raygen))
	{
		NS_ERROR_LOG("Invalid raygen entry!");

		return nullptr;
	}

	auto moduleImpl = std::dynamic_pointer_cast<ModuleImpl>(entry.module);

	if (!moduleImpl)
	{
		NS_ERROR_LOG("Invalid module!");

		return nullptr;
	}

	OptixProgramGroupOptions options = {};
	OptixProgramGroup hProgramGroup = nullptr;
	auto context = moduleImpl->deviceContext();

	OptixProgramGroupDesc desc = {};
	desc.raygen.module = moduleImpl->handle();
	desc.raygen.entryFunctionName = entry.name.c_str();
	desc.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
	desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;

	OptixResult err = optixProgramGroupCreate(context->handle(), &desc, 1, &options, nullptr, nullptr, &hProgramGroup);

	if (err != OPTIX_SUCCESS)
	{
		NS_ERROR_LOG("%s.", optixGetErrorString(err));

		return nullptr;
	}

	return std::make_shared<ProgramImpl>(context, hProgramGroup, Raygen);
}


std::shared_ptr<Program> Program::miss(const ProgramEntry & ms)
{
	if (!ms.module || (queryProgramType(ms.name) != ProgramType::Miss))
	{
		NS_ERROR_LOG("Invalid miss entry!");

		return nullptr;
	}

	auto moduleImpl = std::dynamic_pointer_cast<ModuleImpl>(ms.module);

	if (!moduleImpl)
	{
		NS_ERROR_LOG("Invalid module!");

		return nullptr;
	}

	OptixProgramGroupOptions options = {};
	OptixProgramGroup hProgramGroup = nullptr;
	auto context = moduleImpl->deviceContext();

	OptixProgramGroupDesc desc = {};
	desc.miss.module = moduleImpl->handle();
	desc.miss.entryFunctionName = ms.name.c_str();
	desc.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
	desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;

	OptixResult err = optixProgramGroupCreate(context->handle(), &desc, 1, &options, nullptr, nullptr, &hProgramGroup);

	if (err != OPTIX_SUCCESS)
	{
		NS_ERROR_LOG("%s.", optixGetErrorString(err));

		return nullptr;
	}

	return std::make_shared<ProgramImpl>(context, hProgramGroup, Miss);
}


std::shared_ptr<Program> Program::exception(const ProgramEntry & ex)
{
	if (!ex.module || (queryProgramType(ex.name) != ProgramType::Exception))
	{
		NS_ERROR_LOG("Invalid exception entry!");

		return nullptr;
	}

	auto moduleImpl = std::dynamic_pointer_cast<ModuleImpl>(ex.module);

	if (!moduleImpl)
	{
		NS_ERROR_LOG("Invalid module!");

		return nullptr;
	}

	OptixProgramGroupOptions options = {};
	OptixProgramGroup hProgramGroup = nullptr;
	auto context = moduleImpl->deviceContext();

	OptixProgramGroupDesc desc = {};
	desc.exception.module = moduleImpl->handle();
	desc.exception.entryFunctionName = ex.name.c_str();
	desc.kind = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
	desc.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;

	OptixResult err = optixProgramGroupCreate(context->handle(), &desc, 1, &options, nullptr, nullptr, &hProgramGroup);

	if (err != OPTIX_SUCCESS)
	{
		NS_ERROR_LOG("%s.", optixGetErrorString(err));

		return nullptr;
	}

	return std::make_shared<ProgramImpl>(context, hProgramGroup, Exception);
}


std::shared_ptr<Program> Program::callables(const ProgramEntry & dc, const ProgramEntry & cc)
{
	if (!dc.module && !cc.module)
	{
		NS_ERROR_LOG("At least one callable entry must be valid!");

		return nullptr;
	}
	else if (dc.module && (queryProgramType(dc.name) != ProgramType::DirectCallable))
	{
		NS_ERROR_LOG("Invalid direct callable entry!");

		return nullptr;
	}
	else if (cc.module && (queryProgramType(cc.name) != ProgramType::ContinuationCallable))
	{
		NS_ERROR_LOG("Invalid continuation callable entry!");

		return nullptr;
	}

	std::shared_ptr<DeviceContext> context;

	for (auto & e : { dc, cc })
	{
		if (auto m = std::dynamic_pointer_cast<ModuleImpl>(e.module))
		{
			context = m->deviceContext();

			break;
		}
	}

	if (!context)
	{
		NS_ERROR_LOG("Invalid device context!");

		return nullptr;
	}

	OptixProgramGroup hProgramGroup = nullptr;
	OptixProgramGroupOptions options = {};

	OptixProgramGroupDesc desc = {};
	desc.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
	desc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;

	if (dc.module)
	{
		desc.callables.moduleDC					= std::dynamic_pointer_cast<ModuleImpl>(dc.module)->handle();
		desc.callables.entryFunctionNameDC		= dc.name.c_str();
	}
	if (cc.module)
	{
		desc.callables.moduleCC					= std::dynamic_pointer_cast<ModuleImpl>(cc.module)->handle();
		desc.callables.entryFunctionNameCC		= cc.name.c_str();
	}

	OptixResult err = optixProgramGroupCreate(context->handle(), &desc, 1, &options, nullptr, nullptr, &hProgramGroup);

	if (err != OPTIX_SUCCESS)
	{
		NS_ERROR_LOG("%s.", optixGetErrorString(err));

		return nullptr;
	}

	return std::make_shared<ProgramImpl>(context, hProgramGroup, CallableGroup);
}


std::shared_ptr<Program> Program::hitgroup(const ProgramEntry & is, const ProgramEntry & ah, const ProgramEntry & ch)
{
	if (!is.module && !ah.module && !ch.module)
	{
		NS_ERROR_LOG("At least one hitgroup entry must be valid!");

		return nullptr;
	}
	else if (is.module && (queryProgramType(is.name) != ProgramType::Intersection) && (queryProgramType(is.name) != ProgramType::BuiltinIntersection))
	{
		NS_ERROR_LOG("Invalid intersection entry!");

		return nullptr;
	}
	else if (ah.module && (queryProgramType(ah.name) != ProgramType::AnyHit))
	{
		NS_ERROR_LOG("Invalid any hit entry!");

		return nullptr;
	}
	else if (ch.module && !ch.name.starts_with("__closesthit__"))
	{
		NS_ERROR_LOG("Invalid closest hit entry!");

		return nullptr;
	}

	// Find context from any valid entry
	std::shared_ptr<DeviceContext> context;

	for (auto & e : { is, ah, ch })
	{
		if (auto m = std::dynamic_pointer_cast<ModuleImpl>(e.module))
		{
			context = m->deviceContext();

			break;
		}
	}

	if (!context)
	{
		NS_ERROR_LOG("Invalid device context!");

		return nullptr;
	}

	OptixProgramGroup hProgramGroup = nullptr;
	OptixProgramGroupOptions options = {};

	OptixProgramGroupDesc desc = {};
	desc.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
	desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

	if (is.module)
	{
		desc.hitgroup.moduleIS				= std::dynamic_pointer_cast<ModuleImpl>(is.module)->handle();
		desc.hitgroup.entryFunctionNameIS	= (is.name.starts_with("__builtin_intersection__")) ? nullptr : is.name.c_str();
	}
	if (ah.module)
	{
		desc.hitgroup.moduleAH				= std::dynamic_pointer_cast<ModuleImpl>(ah.module)->handle();
		desc.hitgroup.entryFunctionNameAH	= ah.name.c_str();
	}
	if (ch.module)
	{
		desc.hitgroup.moduleCH				= std::dynamic_pointer_cast<ModuleImpl>(ch.module)->handle();
		desc.hitgroup.entryFunctionNameCH	= ch.name.c_str();
	}

	OptixResult err = optixProgramGroupCreate(context->handle(), &desc, 1, &options, nullptr, nullptr, &hProgramGroup);

	if (err != OPTIX_SUCCESS)
	{
		NS_ERROR_LOG("%s.", optixGetErrorString(err));

		return nullptr;
	}

	return std::make_shared<ProgramImpl>(context, hProgramGroup, HitGroup);
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

Pipeline::Pipeline(SharedContext context, ns::Span<const std::shared_ptr<Program>> programs,
				   const OptixPipelineCompileOptions & pipelineCompileOptions,
				   const OptixPipelineLinkOptions & pipelineLinkOptions)
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

	OptixResult err = optixPipelineCreate(context->handle(), &pipelineCompileOptions, &pipelineLinkOptions, programGroups.data(), (int)programs.size(), nullptr, nullptr, &m_hPipeline);

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