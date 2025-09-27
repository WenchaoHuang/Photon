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

#include <random>
#include <stdlib.h>

#include <nucleus/stream.h>
#include <nucleus/device.h>
#include <nucleus/context.h>
#include <nucleus/array_1d.h>
#include <nucleus/buffer_view.h>
#include <nucleus/scoped_timer.h>

#include <photon/pipeline.h>
#include <photon/accel_struct.h>
#include <photon/device_context.h>
#include "collision_pipeline.ptx.h"
#include "launch_params.h"

/*********************************************************************************
***********************************    main    ***********************************
*********************************************************************************/

int main()
{
	//	host data
	float radius = 1e-2f;
	size_t count = 1000000;
	std::default_random_engine	e;
	std::uniform_real_distribution<float> d;
	std::vector<ns::float3_16a>		leafPos(count);
	std::vector<pt::Aabb>			leafAabb(count);

	for (size_t i = 0; i < leafPos.size(); i++)
	{
		auto p = ns::float3_16a{ d(e), d(e), d(e) };
		leafPos[i] = p;
		leafAabb[i].lower.x = p.x - radius;
		leafAabb[i].lower.y = p.y - radius;
		leafAabb[i].lower.z = p.z - radius;
		leafAabb[i].upper.x = p.x + radius;
		leafAabb[i].upper.y = p.y + radius;
		leafAabb[i].upper.z = p.z + radius;
	}
	
	//	context
	auto device = ns::Context::getInstance()->device(0);
	auto deviceContext = pt::DeviceContext::create(device);
	auto allocator = device->defaultAllocator();
	auto & stream = device->defaultStream();

	//	pipeline
	OptixModuleCompileOptions moduleCompileOptions = {};
	OptixPipelineCompileOptions pipelineCompileOptions = {};
	pipelineCompileOptions.pipelineLaunchParamsVariableName = "launchParams";
	pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	auto module = deviceContext->createModule(collision_pipeline_ptx, moduleCompileOptions, pipelineCompileOptions);
	auto intersectionProg = module->at("__intersection__");
	auto raygenProg = module->at("__raygen__");
	auto missProg = module->at("__miss__");
	auto pipeline = deviceContext->createPipeline({ raygenProg, intersectionProg, missProg }, pipelineCompileOptions);

	//	device data
	ns::Array<int>					devCount(allocator, 1);
	ns::Array<pt::EmptyRecord>		devHitRecord(allocator, 1);
	ns::Array<pt::EmptyRecord>		devMissRecord(allocator, 1);
	ns::Array<pt::EmptyRecord>		devRaygenRecord(allocator, 1);
	ns::Array<LaunchParams>			devLaunchParams(allocator, 1);
	ns::Array<ns::float3_16a>		vertPos(allocator, leafPos.size());
	ns::BufferView1D<pt::Aabb>		aabbBuffer(std::make_shared<ns::Buffer>(allocator, sizeof(pt::Aabb) * leafAabb.size()));
	stream.memcpy(aabbBuffer.data(), leafAabb.data(), leafAabb.size());
	stream.memcpy(vertPos.data(), leafPos.data(), leafPos.size());

	//	accel-struct
	auto accelStruct = deviceContext->createAccelStructAabb();
	pt::AccelStructAabb::BuildInput buildInput;
	buildInput.aabbBuffer = aabbBuffer;
	buildInput.numPrimitives = static_cast<unsigned int>(count);
	accelStruct->build(stream, allocator, buildInput, true, false);

	//	launch parameters
	LaunchParams hostLaunchParams = {};
	hostLaunchParams.radius = radius;
	hostLaunchParams.count = devCount.ptr();
	hostLaunchParams.vertices = vertPos.ptr();
	hostLaunchParams.traversable = accelStruct->handle();

	//	Upload data
	stream.memcpy(devLaunchParams.data(), &hostLaunchParams,  1);
	stream.memcpy(&devMissRecord.data()->header, &missProg->header(), 1);
	stream.memcpy(&devRaygenRecord.data()->header, &raygenProg->header(), 1);
	stream.memcpy(&devHitRecord.data()->header, &intersectionProg->header(), 1);

	//	SBT
	OptixShaderBindingTable sbt = {};
	sbt.raygenRecord = (CUdeviceptr)devRaygenRecord.data();
	sbt.hitgroupRecordBase = (CUdeviceptr)devHitRecord.data();
	sbt.hitgroupRecordStrideInBytes = sizeof(pt::EmptyRecord);
	sbt.hitgroupRecordCount = 1;
	sbt.missRecordBase = (CUdeviceptr)devMissRecord.data();
	sbt.missRecordStrideInBytes = sizeof(pt::EmptyRecord);
	sbt.missRecordCount = 1;

	float timeCost = 0.0f;
	stream.memsetZero(devCount.data(), devCount.bytes()).sync();
	{
		ns::ScopedTimer scopedTimer(stream, [&](float us) { timeCost = us; });

		pipeline->launch<LaunchParams>(stream, devLaunchParams, sbt, count);
	}

	//	Download data
	int hostCount = 0;
	stream.memcpy(&hostCount, devCount.data(), devCount.size()).sync();

	//	Print result
	float guess = 2.0f / 3.0f * 3.14159f * (count * radius) * (count * radius) * radius;
	printf("\n guess count = %d, count = %d, ratio = %f, time = %fus.\n\n", (int)guess, hostCount, hostCount / float(guess), timeCost);

	system("pause");

	return 0;
}