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
#include <nucleus/context.h>
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
	auto device = ns::Context::getInstance()->device(0);
	auto context = pt::SharedContext(device, 4, true);
	auto allocator = device->defaultAllocator();
	auto & stream = device->defaultStream();

	assert(context->device() == device);

	auto devProp = context->properties();
	auto module = context->createModule(rt_program_optixir);

	auto program0 = module->at("");								//	error: empty function name
	auto program1 = module->at("xxxxx");						//	error: invalid function name
	auto program2 = module->at("__raygen__");
	auto program3 = module->at("__raygen__");
	auto program4 = module->at("__raygen__xx");					//	error: not found
	auto program5 = module->at("__exception__");
	auto program6 = module->at("__direct_callable__");
	auto program7 = module->at("__continuation_callable__");
	auto program8 = module->at("__intersection__");
	auto program9 = module->at("__closesthit__");
	auto program10 = module->at("__anyhit__");
	auto program11 = module->at("__miss__");
	auto program12 = pt::Program::combine(program6, program7);
	auto program13 = pt::Program::combine(program8, program9, program10);

	assert(program0 == nullptr);
	assert(program1 == nullptr);
	assert(program2 != nullptr);
	assert(program3 == program2);
	assert(program4 == nullptr);
	assert(program5 != nullptr);
	assert(program6 != nullptr);
	assert(program7 != nullptr);
	assert(program8 != nullptr);
	assert(program9 != nullptr);
	assert(program10 != nullptr);
	assert(program11 != nullptr);

	assert(program2->type() == pt::Program::Raygen);
	assert(program3->type() == pt::Program::Raygen);
	assert(program5->type() == pt::Program::Exception);
	assert(program6->type() == pt::Program::DirectCallable);
	assert(program7->type() == pt::Program::ContinuationCallable);
	assert(program8->type() == pt::Program::Intersection);
	assert(program9->type() == pt::Program::ClosestHit);
	assert(program10->type() == pt::Program::AnyHit);
	assert(program11->type() == pt::Program::Miss);

	ns::Array<LaunchParams>			launchParams(allocator, 1);
	ns::Array<pt::EmptyRecord>		raygenRecord(allocator, 1);
	ns::Array<pt::EmptyRecord>		missRecord(allocator, 1);

	OptixShaderBindingTable sbt = {};
	sbt.raygenRecord = CUdeviceptr(raygenRecord.data());
	sbt.missRecordBase = CUdeviceptr(missRecord.data());
	sbt.missRecordStrideInBytes = sizeof(pt::EmptyRecord);
	sbt.missRecordCount = 1;

	stream.memcpy<void>(missRecord.data(), program11->header().storage, sizeof(pt::SbtHeader));
	stream.memcpy<void>(raygenRecord.data(), program2->header().storage, sizeof(pt::SbtHeader));

	pt::Pipeline pipeline = pt::Pipeline(context, { program2, program9, program11 });
	pipeline.launch<LaunchParams>(stream, launchParams, sbt, 10, 1).sync();
}