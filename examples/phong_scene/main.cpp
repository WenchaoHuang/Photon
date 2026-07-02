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

#include <cmath>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <nucleus/stream.h>
#include <nucleus/device.h>
#include <nucleus/runtime.h>
#include <nucleus/array_1d.h>

#include <photon/pipeline.h>
#include <photon/accel_struct.h>
#include <photon/device_context.h>
#include <photon/sbt_record.h>

#include "launch_params.h"
#include "phong_scene.optixir.h"

#ifdef _MSC_VER
#define EZWIN32_IMPLEMENTATION
	#include "easywin32.h"
#endif

 /*********************************************************************************
 *******************************    Constants    **********************************
 *********************************************************************************/

static constexpr unsigned int IMAGE_WIDTH = 800;
static constexpr unsigned int IMAGE_HEIGHT = 600;

static inline unsigned char toDisplayByte(float value)
{
	value = std::fmax(value, 0.0f);
	value = value / (1.0f + value);
	value = std::pow(value, 1.0f / 2.2f);
	value = std::fmin(value, 1.0f);
	return static_cast<unsigned char>(value * 255.0f);
}

/*********************************************************************************
***************************    Host Vector Math    ********************************
*********************************************************************************/

static inline ns::float3 make_f3(float x, float y, float z) { return ns::float3{ x, y, z }; }
static inline ns::float4 make_f4(float x, float y, float z, float w) { return ns::float4{ x, y, z, w }; }
static inline ns::float3 operator-(ns::float3 a, ns::float3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }

static inline ns::float3 cross(ns::float3 a, ns::float3 b)
{
	return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}

static inline float dot(ns::float3 a, ns::float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

static inline ns::float3 normalize(ns::float3 v)
{
	float len = std::sqrt(dot(v, v));
	return { v.x / len, v.y / len, v.z / len };
}

/*********************************************************************************
*************************    Helper: Build Triangle    ****************************
*********************************************************************************/

//	Creates a ground plane (two triangles forming a quad)
static void buildTriangleGAS(pt::AccelStructTriangle & accelStruct,
							 ns::Stream & stream, ns::AllocPtr allocator,
							 ns::Array<ns::float3> & vertexBuffer,
							 ns::Array<unsigned int> & indexBuffer)
{
	//	Ground plane vertices
	std::vector<ns::float3> vertices = {
		{ -5.0f, -1.0f, -5.0f },
		{  5.0f, -1.0f, -5.0f },
		{  5.0f, -1.0f,  5.0f },
		{ -5.0f, -1.0f,  5.0f },
	};

	std::vector<unsigned int> indices = {
		0, 2, 1,
		0, 3, 2,
	};

	vertexBuffer = ns::Array<ns::float3>(allocator, vertices.size());
	indexBuffer = ns::Array<unsigned int>(allocator, indices.size());
	stream.memcpy(vertexBuffer.data(), vertices.data(), vertices.size());
	stream.memcpy(indexBuffer.data(), indices.data(), indices.size());

	CUdeviceptr vertexPtr = (CUdeviceptr)vertexBuffer.data();
	unsigned int flags = OPTIX_GEOMETRY_FLAG_NONE;

	OptixBuildInputTriangleArray buildInput = {};
	buildInput.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	buildInput.vertexStrideInBytes = sizeof(ns::float3);
	buildInput.numVertices = static_cast<unsigned int>(vertices.size());
	buildInput.vertexBuffers = &vertexPtr;
	buildInput.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	buildInput.indexStrideInBytes = 3 * sizeof(unsigned int);
	buildInput.numIndexTriplets = static_cast<unsigned int>(indices.size() / 3);
	buildInput.indexBuffer = (CUdeviceptr)indexBuffer.data();
	buildInput.flags = &flags;
	buildInput.numSbtRecords = 1;

	pt::AccelStruct::BuildOptions buildOptions = {};

	accelStruct.build(stream, allocator, { buildInput }, buildOptions);
}

/*********************************************************************************
*************************    Helper: Build Spheres    *****************************
*********************************************************************************/

static void buildSphereGAS(pt::AccelStructSphere & accelStruct,
						   ns::Stream & stream, ns::AllocPtr allocator,
						   ns::Array<ns::float3> & centerBuffer, ns::Array<float> & radiusBuffer)
{
	//	Sphere positions
	std::vector<ns::float3> centers = {
		{  0.0f,  0.0f,  0.0f },
		{  2.0f,  0.0f, -1.0f },
		{ -2.0f,  0.5f, -0.5f },
	};

	std::vector<float> radii = { 1.0f, 0.7f, 0.5f };

	centerBuffer = ns::Array<ns::float3>(allocator, centers.size());
	radiusBuffer = ns::Array<float>(allocator, radii.size());
	stream.memcpy(centerBuffer.data(), centers.data(), centers.size());
	stream.memcpy(radiusBuffer.data(), radii.data(), radii.size());

	CUdeviceptr centerPtr = (CUdeviceptr)centerBuffer.data();
	CUdeviceptr radiusPtr = (CUdeviceptr)radiusBuffer.data();
	unsigned int flags = OPTIX_GEOMETRY_FLAG_NONE;

	OptixBuildInputSphereArray buildInput = {};
	buildInput.vertexBuffers = &centerPtr;
	buildInput.radiusBuffers = &radiusPtr;
	buildInput.numVertices = static_cast<unsigned int>(centers.size());
	buildInput.vertexStrideInBytes = sizeof(ns::float3);
	buildInput.radiusStrideInBytes = sizeof(float);
	buildInput.flags = &flags;
	buildInput.numSbtRecords = 1;

	pt::AccelStruct::BuildOptions buildOptions = {};

	accelStruct.build(stream, allocator, { buildInput }, buildOptions);
}

/*********************************************************************************
*************************    Helper: Build Curves    ******************************
*********************************************************************************/

static void buildCurveGAS(pt::AccelStructCurve & accelStruct,
						  ns::Stream & stream, ns::AllocPtr allocator,
						  ns::Array<ns::float4> & controlPointBuffer,
						  ns::Array<float> & curveWidthBuffer,
						  ns::Array<unsigned int> & curveIndexBuffer)
{
	//	Quadratic B-spline curve: 4 control points (x, y, z, width)
	std::vector<ns::float4> controlPoints = {
		{ -3.0f, -1.0f,  2.0f, 0.1f },
		{ -2.0f,  1.0f,  2.0f, 0.12f },
		{ -1.0f, -0.5f,  2.0f, 0.08f },
		{  0.0f,  0.5f,  2.0f, 0.1f },
		{  1.0f,  0.0f,  2.0f, 0.1f },
	};

	//	Segment indices (each segment uses 3 control points for quadratic)
	std::vector<unsigned int> segmentIndices = { 0, 1, 2 };
	std::vector<float> widths(controlPoints.size());
	for (size_t i = 0; i < controlPoints.size(); i++)
	{
		widths[i] = controlPoints[i].w;
	}

	controlPointBuffer = ns::Array<ns::float4>(allocator, controlPoints.size());
	curveWidthBuffer = ns::Array<float>(allocator, widths.size());
	curveIndexBuffer = ns::Array<unsigned int>(allocator, segmentIndices.size());
	stream.memcpy(controlPointBuffer.data(), controlPoints.data(), controlPoints.size());
	stream.memcpy(curveWidthBuffer.data(), widths.data(), widths.size());
	stream.memcpy(curveIndexBuffer.data(), segmentIndices.data(), segmentIndices.size());

	CUdeviceptr controlPointPtr = (CUdeviceptr)controlPointBuffer.data();
	CUdeviceptr widthPtr = (CUdeviceptr)curveWidthBuffer.data();
	unsigned int flags = OPTIX_GEOMETRY_FLAG_NONE;

	OptixBuildInputCurveArray buildInput = {};
	buildInput.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE;
	buildInput.numPrimitives = static_cast<unsigned int>(segmentIndices.size());
	buildInput.vertexBuffers = &controlPointPtr;
	buildInput.numVertices = static_cast<unsigned int>(controlPoints.size());
	buildInput.vertexStrideInBytes = sizeof(ns::float4);
	buildInput.widthBuffers = &widthPtr;
	buildInput.widthStrideInBytes = sizeof(float);
	buildInput.indexBuffer = (CUdeviceptr)curveIndexBuffer.data();
	buildInput.indexStrideInBytes = sizeof(unsigned int);
	buildInput.flag = flags;
	buildInput.primitiveIndexOffset = 0;
	buildInput.endcapFlags = OPTIX_CURVE_ENDCAP_DEFAULT;

	pt::AccelStruct::BuildOptions buildOptions = {};

	accelStruct.build(stream, allocator, { buildInput }, buildOptions);
}

/*********************************************************************************
**************************    Helper: Build AABBs    ******************************
*********************************************************************************/

static void buildAabbGAS(pt::AccelStructAabb & accelStruct,
						 ns::Stream & stream, ns::AllocPtr allocator,
						 ns::Array<pt::Aabb> & aabbBuffer,
						 ns::Array<ns::float3> & aabbCenterBuffer,
						 float & outRadius)
{
	//	Custom spheres represented as AABBs
	float radius = 0.4f;
	outRadius = radius;

	std::vector<ns::float3> centers = {
		{  3.0f,  0.0f,  2.0f },
		{  3.8f,  0.3f,  1.5f },
		{  2.5f, -0.2f,  2.5f },
	};

	std::vector<pt::Aabb> aabbs(centers.size());
	for (size_t i = 0; i < centers.size(); i++)
	{
		aabbs[i].lower.x = centers[i].x - radius;
		aabbs[i].lower.y = centers[i].y - radius;
		aabbs[i].lower.z = centers[i].z - radius;
		aabbs[i].upper.x = centers[i].x + radius;
		aabbs[i].upper.y = centers[i].y + radius;
		aabbs[i].upper.z = centers[i].z + radius;
	}

	aabbBuffer = ns::Array<pt::Aabb>(allocator, aabbs.size());
	aabbCenterBuffer = ns::Array<ns::float3>(allocator, centers.size());
	stream.memcpy(aabbBuffer.data(), aabbs.data(), aabbs.size());
	stream.memcpy(aabbCenterBuffer.data(), centers.data(), centers.size());

	CUdeviceptr aabbPtr = (CUdeviceptr)aabbBuffer.data();
	unsigned int flags = OPTIX_GEOMETRY_FLAG_NONE;

	OptixBuildInputCustomPrimitiveArray buildInput = {};
	buildInput.aabbBuffers = &aabbPtr;
	buildInput.numPrimitives = static_cast<unsigned int>(aabbs.size());
	buildInput.strideInBytes = sizeof(pt::Aabb);
	buildInput.flags = &flags;
	buildInput.numSbtRecords = 1;

	pt::AccelStruct::BuildOptions buildOptions = {};

	accelStruct.build(stream, allocator, { buildInput }, buildOptions);
}

/*********************************************************************************
***********************    Helper: Write PPM Image    *****************************
*********************************************************************************/

static void writePPM(const char * filename, const ns::float3 * image, unsigned int width, unsigned int height)
{
	FILE * fp = nullptr;

#ifdef _MSC_VER
	if (fopen_s(&fp, filename, "wb") != 0)
	{
		fp = nullptr;
	}
#else
	fp = fopen(filename, "wb");
#endif
	if (!fp)
	{
		printf("Error: cannot open %s for writing.\n", filename);
		return;
	}

	fprintf(fp, "P6\n%u %u\n255\n", width, height);

	for (int y = static_cast<int>(height) - 1; y >= 0; y--)
	{
		for (unsigned int x = 0; x < width; x++)
		{
			const ns::float3 c = image[y * width + x];
			unsigned char r = static_cast<unsigned char>(std::fmin(c.x, 1.0f) * 255.0f);
			unsigned char g = static_cast<unsigned char>(std::fmin(c.y, 1.0f) * 255.0f);
			unsigned char b = static_cast<unsigned char>(std::fmin(c.z, 1.0f) * 255.0f);
			fwrite(&r, 1, 1, fp);
			fwrite(&g, 1, 1, fp);
			fwrite(&b, 1, 1, fp);
		}
	}

	fclose(fp);
	printf("Image written to: %s\n", filename);
}

#ifdef _MSC_VER

static std::vector<EzColorRGB> makePreviewImage(const ns::float3 * image, unsigned int width, unsigned int height)
{
	std::vector<EzColorRGB> preview(width * height);

	for (unsigned int y = 0; y < height; y++)
	{
		for (unsigned int x = 0; x < width; x++)
		{
			const ns::float3 c = image[(height - 1 - y) * width + x];
			preview[y * width + x] = EzColorRGB{
				toDisplayByte(c.x),
				toDisplayByte(c.y),
				toDisplayByte(c.z)
			};
		}
	}

	return preview;
}


static void showImageWindow(const EzColorRGB * image, unsigned int width, unsigned int height)
{
	EzWindow window;
	window.open("phong_scene", static_cast<int>(width), static_cast<int>(height));
	window.centerToScreen();
	window.show();
	window.setFocus();

	window.onPaint = [&]()
	{
		window.drawBitmap(image, static_cast<int>(width), static_cast<int>(height));
		return 0;
	};

	window.onClose = [&]()
	{
		window.close();
		return 0;
	};

	window.onKeyboardPress = [&](EzKey key, EzKeyAction action)
	{
		if ((key == EzKey::Escape) && (action == EzKeyAction::Press))
		{
			window.close();
			return 0;
		}

		return 0;
	};

	window.requestRedraw();

	while (window.isOpen())
	{
		window.waitEvent();
	}
}
#endif

/*********************************************************************************
***********************************    main    ***********************************
*********************************************************************************/

int main()
{
	//	Initialize context and device
	auto device = ns::Runtime::device(0);
	auto deviceContext = pt::SharedContext(device);
	auto allocator = device->defaultAllocator();
	auto & stream = device->defaultStream();

	//	========================================================================
	//	Pipeline setup
	//	========================================================================

	OptixModuleCompileOptions moduleCompileOptions = {};
	OptixPipelineCompileOptions pipelineCompileOptions = {};
	pipelineCompileOptions.pipelineLaunchParamsVariableName = "launchParams";
	pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
	pipelineCompileOptions.numPayloadValues = 3;
	pipelineCompileOptions.numAttributeValues = 3;
	pipelineCompileOptions.usesPrimitiveTypeFlags =
		OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE |
		OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE |
		OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE |
		OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

	auto module = deviceContext->createModule(phong_scene_optixir, pipelineCompileOptions, moduleCompileOptions);

	//	Programs
	auto progRaygen = pt::Program::raygen(module->entry("__raygen__"));
	auto progMiss = pt::Program::miss(module->entry("__miss__"));

	//	Hitgroup for triangles (built-in intersection)
	auto progHitTriangle = pt::Program::hitgroup({}, {}, module->entry("__closesthit__triangle"));

	//	Hitgroup for spheres (built-in intersection)
	OptixBuiltinISOptions sphereISOptions = {};
	sphereISOptions.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
	auto sphereIS = deviceContext->getBuiltinISEntry(sphereISOptions, pipelineCompileOptions);
	auto progHitSphere = pt::Program::hitgroup(sphereIS, {}, module->entry("__closesthit__sphere"));

	//	Hitgroup for curves (built-in intersection)
	OptixBuiltinISOptions curveISOptions = {};
	curveISOptions.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE;
	auto curveIS = deviceContext->getBuiltinISEntry(curveISOptions, pipelineCompileOptions);
	auto progHitCurve = pt::Program::hitgroup(curveIS, {}, module->entry("__closesthit__curve"));

	//	Hitgroup for AABB custom primitives (custom intersection)
	auto progHitAabb = pt::Program::hitgroup(module->entry("__intersection__aabb"), {}, module->entry("__closesthit__aabb"));

	pt::Pipeline pipeline(deviceContext,
						  { progRaygen, progMiss, progHitTriangle, progHitSphere, progHitCurve, progHitAabb },
						  pipelineCompileOptions);

	//	========================================================================
	//	Build Geometry Acceleration Structures (GAS)
	//	========================================================================

	//	--- 1. Triangle GAS (ground plane) ---
	ns::Array<ns::float3> triVertices;
	ns::Array<unsigned int> triIndices;
	pt::AccelStructTriangle triangleGAS(deviceContext);
	buildTriangleGAS(triangleGAS, stream, allocator, triVertices, triIndices);

	//	--- 2. Sphere GAS ---
	ns::Array<ns::float3> sphereCenters;
	ns::Array<float> sphereRadii;
	pt::AccelStructSphere sphereGAS(deviceContext);
	buildSphereGAS(sphereGAS, stream, allocator, sphereCenters, sphereRadii);

	//	--- 3. Curve GAS ---
	ns::Array<ns::float4> curveControlPoints;
	ns::Array<float> curveWidths;
	ns::Array<unsigned int> curveIndices;
	pt::AccelStructCurve curveGAS(deviceContext);
	buildCurveGAS(curveGAS, stream, allocator, curveControlPoints, curveWidths, curveIndices);

	//	--- 4. AABB GAS (custom primitives) ---
	ns::Array<pt::Aabb> aabbBuffer;
	ns::Array<ns::float3> aabbCenters;
	float aabbRadius = 0.0f;
	pt::AccelStructAabb aabbGAS(deviceContext);
	buildAabbGAS(aabbGAS, stream, allocator, aabbBuffer, aabbCenters, aabbRadius);

	//	========================================================================
	//	Build Instance Acceleration Structure (IAS) - uses all GAS types
	//	========================================================================

	//	Define instance transforms (identity for simplicity)
	auto makeIdentityTransform = []()
	{
		float transform[12] = {
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
		};
		OptixInstance inst = {};
		memcpy(inst.transform, transform, sizeof(transform));
		inst.visibilityMask = 255;
		return inst;
	};

	std::vector<OptixInstance> instances(4);

	//	Instance 0: Triangle ground plane
	instances[0] = makeIdentityTransform();
	instances[0].instanceId = 0;
	instances[0].sbtOffset = 0;		//	hitgroup index 0
	instances[0].traversableHandle = triangleGAS.handle();

	//	Instance 1: Spheres
	instances[1] = makeIdentityTransform();
	instances[1].instanceId = 1;
	instances[1].sbtOffset = 1;		//	hitgroup index 1
	instances[1].traversableHandle = sphereGAS.handle();

	//	Instance 2: Curves
	instances[2] = makeIdentityTransform();
	instances[2].instanceId = 2;
	instances[2].sbtOffset = 2;		//	hitgroup index 2
	instances[2].traversableHandle = curveGAS.handle();

	//	Instance 3: AABB custom primitives
	instances[3] = makeIdentityTransform();
	instances[3].instanceId = 3;
	instances[3].sbtOffset = 3;		//	hitgroup index 3
	instances[3].traversableHandle = aabbGAS.handle();

	//	Upload instances to device
	ns::Array<OptixInstance> devInstances(allocator, instances.size());
	stream.memcpy(devInstances.data(), instances.data(), instances.size());

	//	Build IAS
	pt::InstAccelStruct ias(deviceContext);
	pt::AccelStruct::BuildOptions iasBuildOptions = {};
	ias.build(stream, allocator, dev::Ptr<const OptixInstance>(devInstances.data(), devInstances.size()),
			  instances.size(), iasBuildOptions);

	//	========================================================================
	//	SBT (Shader Binding Table) setup
	//	========================================================================

	//	Materials
	Material matGround = { { 0.3f, 0.6f, 0.3f }, { 0.2f, 0.2f, 0.2f }, 16.0f };
	Material matSphere = { { 0.8f, 0.2f, 0.2f }, { 1.0f, 1.0f, 1.0f }, 64.0f };
	Material matCurve = { { 0.2f, 0.2f, 0.8f }, { 0.5f, 0.5f, 0.5f }, 32.0f };
	Material matAabb = { { 0.8f, 0.6f, 0.1f }, { 1.0f, 1.0f, 1.0f }, 48.0f };

	//	HitGroup SBT data
	HitGroupData hitDataTriangle = {};
	hitDataTriangle.material = matGround;
	hitDataTriangle.vertices = dev::Ptr<const ns::float3>(triVertices.data(), triVertices.size());
	hitDataTriangle.indices = dev::Ptr<const unsigned int>(triIndices.data(), triIndices.size());

	HitGroupData hitDataSphere = {};
	hitDataSphere.material = matSphere;
	hitDataSphere.sphereCenters = dev::Ptr<const ns::float3>(sphereCenters.data(), sphereCenters.size());
	hitDataSphere.sphereRadii = dev::Ptr<const float>(sphereRadii.data(), sphereRadii.size());

	HitGroupData hitDataCurve = {};
	hitDataCurve.material = matCurve;
	hitDataCurve.curveControlPoints = dev::Ptr<const ns::float4>(curveControlPoints.data(), curveControlPoints.size());

	HitGroupData hitDataAabb = {};
	hitDataAabb.material = matAabb;
	hitDataAabb.aabbCenters = dev::Ptr<const ns::float3>(aabbCenters.data(), aabbCenters.size());
	hitDataAabb.aabbRadius = aabbRadius;

	//	Allocate SBT records on device
	ns::Array<pt::SbtRecord<>>				devRaygenRecord(allocator, 1);
	ns::Array<pt::SbtRecord<>>				devMissRecord(allocator, 1);
	ns::Array<pt::SbtRecord<HitGroupData>>	devHitRecords(allocator, 4);

	//	Upload raygen & miss SBT headers
	stream.memcpy(&devRaygenRecord.data()->header, &progRaygen->header(), 1);
	stream.memcpy(&devMissRecord.data()->header, &progMiss->header(), 1);

	//	Upload hitgroup SBT records
	pt::SbtRecord<HitGroupData> hitRecords[4];
	memcpy(&hitRecords[0].header, &progHitTriangle->header(), sizeof(pt::SbtHeader));
	hitRecords[0].data = hitDataTriangle;
	memcpy(&hitRecords[1].header, &progHitSphere->header(), sizeof(pt::SbtHeader));
	hitRecords[1].data = hitDataSphere;
	memcpy(&hitRecords[2].header, &progHitCurve->header(), sizeof(pt::SbtHeader));
	hitRecords[2].data = hitDataCurve;
	memcpy(&hitRecords[3].header, &progHitAabb->header(), sizeof(pt::SbtHeader));
	hitRecords[3].data = hitDataAabb;

	stream.memcpy(devHitRecords.data(), hitRecords, 4);

	//	Fill SBT struct
	OptixShaderBindingTable sbt = {};
	sbt.raygenRecord = (CUdeviceptr)devRaygenRecord.data();
	sbt.missRecordBase = (CUdeviceptr)devMissRecord.data();
	sbt.missRecordStrideInBytes = sizeof(pt::SbtRecord<>);
	sbt.missRecordCount = 1;
	sbt.hitgroupRecordBase = (CUdeviceptr)devHitRecords.data();
	sbt.hitgroupRecordStrideInBytes = sizeof(pt::SbtRecord<HitGroupData>);
	sbt.hitgroupRecordCount = 4;

	//	========================================================================
	//	Launch parameters
	//	========================================================================

	LaunchParams hostParams = {};
	hostParams.width = IMAGE_WIDTH;
	hostParams.height = IMAGE_HEIGHT;

	//	Camera setup (look at origin from slightly above)
	hostParams.camPos = { 0.0f, 2.0f, 8.0f };
	ns::float3 lookAt = { 0.0f, 0.0f, 0.0f };
	ns::float3 up = { 0.0f, 1.0f, 0.0f };
	ns::float3 w = normalize(hostParams.camPos - lookAt);	//	camera forward (away from scene)
	ns::float3 u = normalize(cross(up, w));					//	camera right
	ns::float3 v = cross(w, u);								//	camera up

	float fovY = 45.0f * 3.14159265f / 180.0f;
	float aspect = float(IMAGE_WIDTH) / float(IMAGE_HEIGHT);
	float halfH = std::tan(fovY * 0.5f);
	float halfW = halfH * aspect;

	hostParams.camU = { halfW * u.x, halfW * u.y, halfW * u.z };
	hostParams.camV = { halfH * v.x, halfH * v.y, halfH * v.z };
	hostParams.camW = { -w.x, -w.y, -w.z };

	//	Light
	hostParams.lightPos = { 5.0f, 10.0f, 5.0f };
	hostParams.lightColor = { 1.0f, 1.0f, 0.95f };
	hostParams.ambientColor = { 0.1f, 0.1f, 0.15f };

	//	Scene traversable
	hostParams.traversable = ias.handle();

	//	Allocate output image
	ns::Array<ns::float3> devImage(allocator, IMAGE_WIDTH * IMAGE_HEIGHT);
	hostParams.image = dev::Ptr<ns::float3>(devImage.data(), devImage.size());

	//	Upload launch params
	ns::Array<LaunchParams> devLaunchParams(allocator, 1);
	stream.memcpy(devLaunchParams.data(), &hostParams, 1);

	//	========================================================================
	//	Launch
	//	========================================================================

	pipeline.launch<LaunchParams>(stream, devLaunchParams, sbt, IMAGE_WIDTH, IMAGE_HEIGHT);
	stream.sync();

	//	========================================================================
	//	Download and save image
	//	========================================================================

	std::vector<ns::float3> hostImage(IMAGE_WIDTH * IMAGE_HEIGHT);
	stream.memcpy(hostImage.data(), devImage.data(), devImage.size()).sync();

#ifdef _MSC_VER
	auto previewImage = makePreviewImage(hostImage.data(), IMAGE_WIDTH, IMAGE_HEIGHT);
	showImageWindow(previewImage.data(), IMAGE_WIDTH, IMAGE_HEIGHT);
#else
	writePPM("phong_scene.ppm", hostImage.data(), IMAGE_WIDTH, IMAGE_HEIGHT);
#endif

	printf("\nPhong scene rendered successfully!\n");
	printf("Scene contains:\n");
	printf("  - AccelStructTriangle: ground plane\n");
	printf("  - AccelStructSphere:   3 spheres\n");
	printf("  - AccelStructCurve:    quadratic B-spline curve\n");
	printf("  - AccelStructAabb:     3 custom sphere primitives\n");
	printf("  - InstAccelStruct:     top-level IAS combining all GAS\n");

	return 0;
}