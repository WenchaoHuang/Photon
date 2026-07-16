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

#include <nucleus/stream.h>
#include <nucleus/device.h>
#include <nucleus/runtime.h>
#include <nucleus/array_1d.h>

#include <photon/pipeline.h>
#include <photon/device_context.h>

#include "launch_params.h"
#include "rt_program.optixir.h"

/*********************************************************************************
******************************    pipeline_test    *******************************
*********************************************************************************/

void pipeline_test()
{
	auto device = ns::Runtime::device(0);
	auto context = pt::DeviceContext::create(device, 4, true);
	auto allocator = device->defaultAllocator();
	auto & stream = device->defaultStream();

	assert(context->device() == device);

	OptixPipelineCompileOptions pipelineCompileOptions = {};
	pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;

	auto devProp = context->properties();
	auto module = context->createModule(rt_program_optixir, pipelineCompileOptions);

	auto entry0 = module->entry("");							//	error: empty function name
	auto entry1 = module->entry("xxxxx");						//	error: invalid function name
	auto entry_rg = module->entry("__raygen__");
	auto entry_ex = module->entry("__exception__");
	auto entry_dc = module->entry("__direct_callable__");
	auto entry_cc = module->entry("__continuation_callable__");
	auto entry_is = module->entry("__intersection__");
	auto entry_ch = module->entry("__closesthit__");
	auto entry_ah = module->entry("__anyhit__");
	auto entry_ms = module->entry("__miss__");

	assert(!entry0.module);
	assert(!entry1.module);
	assert(entry_rg.module);
	assert(entry_ch.module);
	assert(entry_ms.module);

	auto program_ms = pt::Program::miss(entry_ms);
	auto program_rg = pt::Program::raygen(entry_rg);
	auto program_ex = pt::Program::exception(entry_ex);
	auto program_hg = pt::Program::hitgroup(entry_is, entry_ah, entry_ch);
	auto program_cg = pt::Program::callables(entry_dc, entry_cc);
	auto program_ch = pt::Program::hitgroup({}, {}, entry_ch);

	OptixBuiltinISOptions builtinISOptions = {};
	builtinISOptions.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
	auto builtin_is = context->getBuiltinISEntry(builtinISOptions, pipelineCompileOptions);
	auto program_bi = pt::Program::hitgroup(builtin_is, {}, entry_ch);

	assert(builtin_is.module);
	assert(program_rg != nullptr);
	assert(program_ms != nullptr);
	assert(program_ex != nullptr);
	assert(program_hg != nullptr);
	assert(program_cg != nullptr);
	assert(program_ch != nullptr);
	assert(program_bi != nullptr);

	ns::Array<LaunchParams>			launchParams(allocator, 1);
	ns::Array<pt::SbtRecord<>>		raygenRecord(allocator, 1);
	ns::Array<pt::SbtRecord<>>		missRecord(allocator, 1);

	OptixShaderBindingTable sbt = {};
	sbt.raygenRecord = CUdeviceptr(raygenRecord.data());
	sbt.missRecordBase = CUdeviceptr(missRecord.data());
	sbt.missRecordStrideInBytes = sizeof(pt::SbtRecord<>);
	sbt.missRecordCount = 1;

	stream.memcpy<void>(missRecord.data(), program_ms->header().storage, sizeof(pt::SbtRecord<>));
	stream.memcpy<void>(raygenRecord.data(), program_rg->header().storage, sizeof(pt::SbtRecord<>));

	pt::Pipeline pipeline = pt::Pipeline(context, { program_rg, program_ch, program_ms, program_bi }, pipelineCompileOptions);
	pipeline.launch<LaunchParams>(stream, launchParams, sbt, 10, 1).sync();
}