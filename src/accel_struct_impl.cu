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

#include "accel_struct_impl.h"
#include <nucleus/launch_utils.cuh>
#include <optix_stubs.h>

PHOTON_USING_NAMESPACE

/*********************************************************************************
*******************************    Validations    ********************************
*********************************************************************************/

static_assert(static_cast<int>(GeomAccelStruct::GeomFlags::None)								== OPTIX_GEOMETRY_FLAG_NONE);
static_assert(static_cast<int>(GeomAccelStruct::GeomFlags::DisableAnyhit)						== OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);
static_assert(static_cast<int>(GeomAccelStruct::GeomFlags::RequireSingleAnyhitCall)				== OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL);
#if OPTIX_VERSION >= 70500
static_assert(static_cast<int>(GeomAccelStruct::GeomFlags::DisableTriangleFaceCulling)			== OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING);
#endif

#if OPTIX_VERSION >= 70100
static_assert(static_cast<int>(AccelStructCurve::CurveType::RoundLinear)						== OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR);
static_assert(static_cast<int>(AccelStructCurve::CurveType::RoundCubicBSpline)					== OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE);
static_assert(static_cast<int>(AccelStructCurve::CurveType::RoundQuadraticBSpline)				== OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE);
#endif
#if OPTIX_VERSION >= 70400
static_assert(static_cast<int>(AccelStructCurve::CurveType::RoundCatmullRom)					== OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM);
#endif
#if OPTIX_VERSION >= 70700
static_assert(static_cast<int>(AccelStructCurve::CurveType::RoundCubicBezier)					== OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER);
static_assert(static_cast<int>(AccelStructCurve::CurveType::FlatQuadraticBSpline)				== OPTIX_PRIMITIVE_TYPE_FLAT_QUADRATIC_BSPLINE);
#endif

static_assert(static_cast<int>(InstAccelStruct::InstFlags::None)								== OPTIX_INSTANCE_FLAG_NONE);
static_assert(static_cast<int>(InstAccelStruct::InstFlags::DisableAnyhit)						== OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT);
static_assert(static_cast<int>(InstAccelStruct::InstFlags::EnforceAnyhit)						== OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT);
static_assert(static_cast<int>(InstAccelStruct::InstFlags::eFlagFlipTriangleFacing)				== OPTIX_INSTANCE_FLAG_FLIP_TRIANGLE_FACING);
static_assert(static_cast<int>(InstAccelStruct::InstFlags::DisableTriangleFaceCulling)			== OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING);
#if OPTIX_VERSION >= 70600
static_assert(static_cast<int>(InstAccelStruct::InstFlags::DisableOpacityMicromaps)				== OPTIX_INSTANCE_FLAG_DISABLE_OPACITY_MICROMAPS);
static_assert(static_cast<int>(InstAccelStruct::InstFlags::ForceOpacityMicromapAsTwoState)		== OPTIX_INSTANCE_FLAG_FORCE_OPACITY_MICROMAP_2_STATE);
#endif

/*********************************************************************************
*****************************    AccelStructBase    ******************************
*********************************************************************************/

AccelStructBase::AccelStructBase(std::shared_ptr<DeviceContextImpl> deviceContext) : m_deviceContext(deviceContext), m_hTraversable(0), m_buildOptions(OptixAccelBuildOptions{}), m_numSbtRecords(0)
{

}


void AccelStructBase::build(ns::Stream & stream, ns::AllocPtr allocator, const std::vector<OptixBuildInput> & buildInputs, OptixAccelBuildOptions buildOptions)
{
	OptixAccelBufferSizes accelBufferSizes = {};

	buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixResult err = optixAccelComputeMemoryUsage(m_deviceContext->handle(), &buildOptions, buildInputs.data(), (uint32_t)buildInputs.size(), &accelBufferSizes);

	if (err == OPTIX_SUCCESS)
	{
		m_outputBuffer.resize(allocator, accelBufferSizes.outputSizeInBytes);

		m_tempBuffer.resize(allocator, NS_MAX(accelBufferSizes.tempSizeInBytes, accelBufferSizes.tempUpdateSizeInBytes) + 15);		//!	last aligned 8-bytes for storing compacted size.

		if (buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION)
		{
			OptixAccelEmitDesc			emittedProp = {};
			OptixTraversableHandle		outputHandle = 0;
			emittedProp.type			= OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
			emittedProp.result			= CUdeviceptr(m_tempBuffer.data() + ((m_tempBuffer.size() - 15 + 7) / 8) * 8);

			err = optixAccelBuild(m_deviceContext->handle(), stream.handle(), &buildOptions, buildInputs.data(), (uint32_t)buildInputs.size(),
									  (CUdeviceptr)m_tempBuffer.data(), m_tempBuffer.bytes(), (CUdeviceptr)m_outputBuffer.data(), m_outputBuffer.bytes(), &outputHandle, &emittedProp, 1);

			if (err == OPTIX_SUCCESS)
			{
				uint64_t compactedSize = 0;

				stream.memcpy<void>(&compactedSize, (const void*)emittedProp.result, sizeof(uint64_t));
				stream.sync();

				m_compactedBuffer.resize(allocator, compactedSize);

				err = optixAccelCompact(m_deviceContext->handle(), stream.handle(), outputHandle, (CUdeviceptr)m_compactedBuffer.data(), m_compactedBuffer.bytes(), &m_hTraversable);
			}
		}
		else
		{
			err = optixAccelBuild(m_deviceContext->handle(), stream.handle(), &buildOptions, buildInputs.data(), (uint32_t)buildInputs.size(),
									  (CUdeviceptr)m_tempBuffer.data(), m_tempBuffer.bytes(), (CUdeviceptr)m_outputBuffer.data(), m_outputBuffer.bytes(), &m_hTraversable, nullptr, 0);
		}
	}

	if (err != OPTIX_SUCCESS)
	{
		NS_ERROR_LOG("Failed to build acceleration structure: %s.", optixGetErrorString(err));

		throw err;
	}

	m_buildOptions = buildOptions;
	m_buildInputs = buildInputs;
}


void AccelStructBase::rebuild(ns::Stream & stream)
{
	if (m_hTraversable != 0)
	{
		OptixTraversableHandle outputHandle = 0;

		m_buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

		OptixResult err = optixAccelBuild(m_deviceContext->handle(), stream.handle(), &m_buildOptions, m_buildInputs.data(), (uint32_t)m_buildInputs.size(),
											  (CUdeviceptr)m_tempBuffer.data(), m_tempBuffer.bytes(), (CUdeviceptr)m_outputBuffer.data(), m_outputBuffer.bytes(), &outputHandle, nullptr, 0);

		if (err == OPTIX_SUCCESS)
		{
			if (this->allowCompaction())
			{
				err = optixAccelCompact(m_deviceContext->handle(), stream.handle(), outputHandle, (CUdeviceptr)m_compactedBuffer.data(), m_compactedBuffer.bytes(), &m_hTraversable);
			}
			else
			{
				m_hTraversable = outputHandle;
			}
		}

		if (err != OPTIX_SUCCESS)
		{
			NS_ERROR_LOG("Failed to rebuild acceleration structure: %s.", optixGetErrorString(err));

			throw err;
		}
	}
}


void AccelStructBase::refit(ns::Stream & stream)
{
	OptixResult err = OPTIX_SUCCESS;

	if (this->allowUpdate() && (m_hTraversable != 0))
	{
		m_buildOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;

		if (this->allowCompaction())
		{
			err = optixAccelBuild(m_deviceContext->handle(), stream.handle(), &m_buildOptions, m_buildInputs.data(), (uint32_t)m_buildInputs.size(),
									  (CUdeviceptr)m_tempBuffer.data(), m_tempBuffer.bytes(), (CUdeviceptr)m_compactedBuffer.data(), m_compactedBuffer.bytes(), &m_hTraversable, nullptr, 0);
		}
		else
		{
			err = optixAccelBuild(m_deviceContext->handle(), stream.handle(), &m_buildOptions, m_buildInputs.data(), (uint32_t)m_buildInputs.size(),
									  (CUdeviceptr)m_tempBuffer.data(), m_tempBuffer.bytes(), (CUdeviceptr)m_outputBuffer.data(), m_outputBuffer.bytes(), &m_hTraversable, nullptr, 0);
		}

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


AccelStructBase::~AccelStructBase()
{

}

/*********************************************************************************
*************************    AccelStructTriangleImpl    **************************
*********************************************************************************/

void AccelStructTriangleImpl::build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<BuildInput> buildInputs, bool preferFastTrace, bool allowUpdate)
{
	m_numSbtRecords = 0;
	m_geomFlags.resize(buildInputs.size());
	m_vertBuffers.resize(buildInputs.size());
	m_buildInputs.resize(buildInputs.size());

	std::vector<OptixBuildInput> optixBuildInputs(buildInputs.size());

	for (size_t i = 0; i < optixBuildInputs.size(); i++)
	{
		const bool useInexBuffer = (buildInputs[i].indexBuffer != nullptr) && (buildInputs[i].numIndexTriplets > 0);

		if (buildInputs[i].perSbtRecordFlags.empty())
		{
			m_geomFlags[i].assign(buildInputs[i].numSbtRecords, OPTIX_GEOMETRY_FLAG_NONE);
		}
		else if (buildInputs[i].perSbtRecordFlags.size() == buildInputs[i].numSbtRecords)
		{
			m_geomFlags[i].resize(buildInputs[i].perSbtRecordFlags.size());

			std::memcpy(m_geomFlags[i].data(),buildInputs[i].perSbtRecordFlags.data(), sizeof(GeomFlags) * buildInputs[i].perSbtRecordFlags.size());
		}
		else
		{
			NS_ASSERT_LOG_IF(buildInputs[i].perSbtRecordFlags.size() != buildInputs[i].numSbtRecords, "Geometry flags does not match with numSbtRecords!");

			return;
		}

		m_buildInputs[i]													= buildInputs[i];
		m_vertBuffers[i]													= (CUdeviceptr)buildInputs[i].vertexBuffer.data();
		m_numSbtRecords														+= buildInputs[i].numSbtRecords;
		optixBuildInputs[i].type											= OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
		optixBuildInputs[i].triangleArray.flags								= m_geomFlags[i].data();
		optixBuildInputs[i].triangleArray.vertexFormat						= OPTIX_VERTEX_FORMAT_FLOAT3;
		optixBuildInputs[i].triangleArray.vertexStrideInBytes				= sizeof(ns::float3_16a);
		optixBuildInputs[i].triangleArray.vertexBuffers						= &m_vertBuffers[i];
		optixBuildInputs[i].triangleArray.numVertices						= buildInputs[i].numVertices;
		optixBuildInputs[i].triangleArray.indexBuffer						= useInexBuffer ? (CUdeviceptr)buildInputs[i].indexBuffer.data() : NULL;
		optixBuildInputs[i].triangleArray.numIndexTriplets					= useInexBuffer ? buildInputs[i].numIndexTriplets : 0;
		optixBuildInputs[i].triangleArray.indexStrideInBytes				= useInexBuffer ? sizeof(ns::int3_16a) : 0;
		optixBuildInputs[i].triangleArray.preTransform						= NULL;
		optixBuildInputs[i].triangleArray.numSbtRecords						= buildInputs[i].numSbtRecords;
		optixBuildInputs[i].triangleArray.primitiveIndexOffset				= 0;
		optixBuildInputs[i].triangleArray.sbtIndexOffsetBuffer				= (CUdeviceptr)buildInputs[i].sbtIndexOffsetBuffer.data();
		optixBuildInputs[i].triangleArray.sbtIndexOffsetSizeInBytes			= sizeof(uint32_t);
		optixBuildInputs[i].triangleArray.sbtIndexOffsetStrideInBytes		= sizeof(uint32_t);
	#if OPTIX_VERSION >= 70100
		optixBuildInputs[i].triangleArray.indexFormat						= useInexBuffer ? OPTIX_INDICES_FORMAT_UNSIGNED_INT3 : OPTIX_INDICES_FORMAT_NONE;
		optixBuildInputs[i].triangleArray.transformFormat					= OPTIX_TRANSFORM_FORMAT_NONE;
	#else
		optixBuildInputs[i].triangleArray.indexFormat						= OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	#endif
	}

	OptixAccelBuildOptions						buildOptions = {};
	buildOptions.operation						= OPTIX_BUILD_OPERATION_BUILD;
	buildOptions.buildFlags						= OPTIX_BUILD_FLAG_NONE;
//	buildOptions.buildFlags						|= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	buildOptions.buildFlags						|= preferFastTrace ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE : OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
	buildOptions.buildFlags						|= allowUpdate ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0;
	buildOptions.motionOptions.numKeys			= 0;
	buildOptions.motionOptions.timeBegin		= 0.0f;
	buildOptions.motionOptions.timeEnd			= 0.0f;
	buildOptions.motionOptions.flags			= OPTIX_MOTION_FLAG_NONE;

	AccelStructBase::build(stream, allocator, optixBuildInputs, buildOptions);
}

/*********************************************************************************
***************************    AccelStructAabbImpl    ****************************
*********************************************************************************/

void AccelStructAabbImpl::build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<BuildInput> buildInputs, bool preferFastTrace, bool allowUpdate)
{
	m_numSbtRecords = 0;
	m_geomFlags.resize(buildInputs.size());
	m_aabbBuffers.resize(buildInputs.size());
	m_buildInputs.resize(buildInputs.size());

	std::vector<OptixBuildInput> optixBuildInputs(buildInputs.size());

	for (size_t i = 0; i < optixBuildInputs.size(); i++)
	{
		if (buildInputs[i].perSbtRecordFlags.empty())
		{
			m_geomFlags[i].assign(buildInputs[i].numSbtRecords, OPTIX_GEOMETRY_FLAG_NONE);
		}
		else if (buildInputs[i].perSbtRecordFlags.size() == buildInputs[i].numSbtRecords)
		{
			m_geomFlags[i].resize(buildInputs[i].perSbtRecordFlags.size());

			std::memcpy(m_geomFlags[i].data(),buildInputs[i].perSbtRecordFlags.data(), sizeof(GeomFlags) * buildInputs[i].perSbtRecordFlags.size());
		}
		else
		{
			NS_ASSERT_LOG_IF(buildInputs[i].perSbtRecordFlags.size() != buildInputs[i].numSbtRecords, "Geometry flags does not match with numSbtRecords!");

			return;
		}

		m_buildInputs[i]															= buildInputs[i];
		m_aabbBuffers[i]															= (CUdeviceptr)buildInputs[i].aabbBuffer.data();
		m_numSbtRecords																+= buildInputs[i].numSbtRecords;
		optixBuildInputs[i].type													= OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
	#if OPTIX_VERSION >= 70100
		optixBuildInputs[i].customPrimitiveArray.flags								= m_geomFlags[i].data();
		optixBuildInputs[i].customPrimitiveArray.aabbBuffers						= &m_aabbBuffers[i];
		optixBuildInputs[i].customPrimitiveArray.strideInBytes						= sizeof(Aabb);
		optixBuildInputs[i].customPrimitiveArray.numPrimitives						= buildInputs[i].numPrimitives;
		optixBuildInputs[i].customPrimitiveArray.numSbtRecords						= buildInputs[i].numSbtRecords;
		optixBuildInputs[i].customPrimitiveArray.primitiveIndexOffset				= 0;
		optixBuildInputs[i].customPrimitiveArray.sbtIndexOffsetBuffer				= (CUdeviceptr)buildInputs[i].sbtIndexOffsetBuffer.data();
		optixBuildInputs[i].customPrimitiveArray.sbtIndexOffsetSizeInBytes			= sizeof(uint32_t);
		optixBuildInputs[i].customPrimitiveArray.sbtIndexOffsetStrideInBytes		= sizeof(uint32_t);
	#else
		optixBuildInputs[i].aabbArray.flags											= m_geomFlags[i].data();
		optixBuildInputs[i].aabbArray.aabbBuffers									= &m_aabbBuffers[i];
		optixBuildInputs[i].aabbArray.strideInBytes									= sizeof(Aabb);
		optixBuildInputs[i].aabbArray.numPrimitives									= buildInputs[i].numPrimitives;
		optixBuildInputs[i].aabbArray.numSbtRecords									= buildInputs[i].numSbtRecords;
		optixBuildInputs[i].aabbArray.primitiveIndexOffset							= 0;
		optixBuildInputs[i].aabbArray.sbtIndexOffsetBuffer							= (CUdeviceptr)buildInputs[i].sbtIndexOffsetBuffer.data();
		optixBuildInputs[i].aabbArray.sbtIndexOffsetSizeInBytes						= sizeof(uint32_t);
		optixBuildInputs[i].aabbArray.sbtIndexOffsetStrideInBytes					= sizeof(uint32_t);
	#endif
	}

	OptixAccelBuildOptions						buildOptions = {};
	buildOptions.operation						= OPTIX_BUILD_OPERATION_BUILD;
	buildOptions.buildFlags						= OPTIX_BUILD_FLAG_NONE;
//	buildOptions.buildFlags						|= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	buildOptions.buildFlags						|= preferFastTrace ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE : OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
	buildOptions.buildFlags						|= allowUpdate ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0;
	buildOptions.motionOptions.numKeys			= 0;
	buildOptions.motionOptions.timeBegin		= 0.0f;
	buildOptions.motionOptions.timeEnd			= 0.0f;
	buildOptions.motionOptions.flags			= OPTIX_MOTION_FLAG_NONE;

	AccelStructBase::build(stream, allocator, optixBuildInputs, buildOptions);
}

/*********************************************************************************
***************************    AccelStructCurveImpl    ***************************
*********************************************************************************/

void AccelStructCurveImpl::build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<BuildInput> buildInputs, bool preferFastTrace, bool allowUpdate)
{
	m_buildInputs.resize(buildInputs.size());
	m_vertBuffers.resize(buildInputs.size());
	m_widthBuffers.resize(buildInputs.size());
	m_numSbtRecords = static_cast<uint32_t>(buildInputs.size());

	std::vector<OptixBuildInput> optixBuildInputs(buildInputs.size());

	for (size_t i = 0; i < buildInputs.size(); i++)
	{
		m_buildInputs[i]										= buildInputs[i];
		m_vertBuffers[i]										= (CUdeviceptr)buildInputs[i].vertexBuffer.data();
		m_widthBuffers[i]										= (CUdeviceptr)buildInputs[i].widthBuffer.data();

	#if OPTIX_VERSION >= 70400
		optixBuildInputs[i].curveArray.endcapFlags				= OPTIX_CURVE_ENDCAP_DEFAULT;
	#endif
	#if OPTIX_VERSION >= 70100
		optixBuildInputs[i].type								= OPTIX_BUILD_INPUT_TYPE_CURVES;
		optixBuildInputs[i].curveArray.flag						= buildInputs[i].flags;
		optixBuildInputs[i].curveArray.curveType				= static_cast<OptixPrimitiveType>(buildInputs[i].curveType);
		optixBuildInputs[i].curveArray.numVertices				= buildInputs[i].numVertices;
		optixBuildInputs[i].curveArray.numPrimitives			= buildInputs[i].numPrimitives;
		optixBuildInputs[i].curveArray.vertexBuffers			= &m_vertBuffers[i];
		optixBuildInputs[i].curveArray.vertexStrideInBytes		= sizeof(ns::float3_16a);
		optixBuildInputs[i].curveArray.indexBuffer				= (CUdeviceptr)buildInputs[i].indexBuffer.data();
		optixBuildInputs[i].curveArray.indexStrideInBytes		= sizeof(uint32_t);
		optixBuildInputs[i].curveArray.widthBuffers				= &m_widthBuffers[i];
		optixBuildInputs[i].curveArray.widthStrideInBytes		= sizeof(float);
		optixBuildInputs[i].curveArray.normalBuffers			= nullptr;
		optixBuildInputs[i].curveArray.normalStrideInBytes		= 0;
		optixBuildInputs[i].curveArray.primitiveIndexOffset		= 0;
	#endif
	}

	OptixAccelBuildOptions						buildOptions = {};
	buildOptions.operation						= OPTIX_BUILD_OPERATION_BUILD;
	buildOptions.buildFlags						= OPTIX_BUILD_FLAG_NONE;
//	buildOptions.buildFlags						|= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	buildOptions.buildFlags						|= preferFastTrace ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE : OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
	buildOptions.buildFlags						|= allowUpdate ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0;
	buildOptions.motionOptions.numKeys			= 0;
	buildOptions.motionOptions.timeBegin		= 0.0f;
	buildOptions.motionOptions.timeEnd			= 0.0f;
	buildOptions.motionOptions.flags			= OPTIX_MOTION_FLAG_NONE;

	AccelStructBase::build(stream, allocator, optixBuildInputs, buildOptions);
}

/*********************************************************************************
**************************    AccelStructSphereImpl    ***************************
*********************************************************************************/

void AccelStructSphereImpl::build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<BuildInput> buildInputs, bool preferFastTrace, bool allowUpdate)
{
	m_numSbtRecords = 0;
	m_geomFlags.resize(buildInputs.size());
	m_buildInputs.resize(buildInputs.size());
	m_vertBuffers.resize(buildInputs.size());
	m_radiusBuffers.resize(buildInputs.size());

	std::vector<OptixBuildInput> optixBuildInputs(buildInputs.size());
	
	for (size_t i = 0; i < buildInputs.size(); i++)
	{
		if (buildInputs[i].perSbtRecordFlags.empty())
		{
			m_geomFlags[i].assign(buildInputs[i].numSbtRecords, OPTIX_GEOMETRY_FLAG_NONE);
		}
		else if (buildInputs[i].perSbtRecordFlags.size() == buildInputs[i].numSbtRecords)
		{
			m_geomFlags[i].resize(buildInputs[i].perSbtRecordFlags.size());

			std::memcpy(m_geomFlags[i].data(),buildInputs[i].perSbtRecordFlags.data(), sizeof(GeomFlags) * buildInputs[i].perSbtRecordFlags.size());
		}
		else
		{
			NS_ASSERT_LOG_IF(buildInputs[i].perSbtRecordFlags.size() != buildInputs[i].numSbtRecords, "Geometry flags does not match with numSbtRecords!");

			return;
		}

		m_buildInputs[i]												= buildInputs[i];
		m_vertBuffers[i]												= (CUdeviceptr)buildInputs[i].vertexBuffer.data();
		m_radiusBuffers[i]												= (CUdeviceptr)buildInputs[i].radiusBuffer.data();
		m_numSbtRecords													+= buildInputs[i].numSbtRecords;
	#if OPTIX_VERSION >= 70500
		optixBuildInputs[i].type										= OPTIX_BUILD_INPUT_TYPE_SPHERES;
		optixBuildInputs[i].sphereArray.flags							= m_geomFlags[i].data();
		optixBuildInputs[i].sphereArray.numVertices						= buildInputs[i].numVertices;
		optixBuildInputs[i].sphereArray.vertexBuffers					= &m_vertBuffers[i];
		optixBuildInputs[i].sphereArray.radiusBuffers					= &m_radiusBuffers[i];
		optixBuildInputs[i].sphereArray.radiusStrideInBytes				= sizeof(float);
		optixBuildInputs[i].sphereArray.singleRadius					= buildInputs[i].singleRadius;
		optixBuildInputs[i].sphereArray.numSbtRecords					= buildInputs[i].numSbtRecords;
		optixBuildInputs[i].sphereArray.primitiveIndexOffset			= 0;
		optixBuildInputs[i].sphereArray.sbtIndexOffsetBuffer			= (CUdeviceptr)buildInputs[i].sbtIndexOffsetBuffer.data();
		optixBuildInputs[i].sphereArray.sbtIndexOffsetSizeInBytes		= sizeof(uint32_t);
		optixBuildInputs[i].sphereArray.sbtIndexOffsetStrideInBytes		= sizeof(uint32_t);
	#endif
	}

	OptixAccelBuildOptions						buildOptions = {};
	buildOptions.operation						= OPTIX_BUILD_OPERATION_BUILD;
	buildOptions.buildFlags						= OPTIX_BUILD_FLAG_NONE;
//	buildOptions.buildFlags						|= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	buildOptions.buildFlags						|= preferFastTrace ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE : OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
	buildOptions.buildFlags						|= allowUpdate ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0;
	buildOptions.motionOptions.numKeys			= 0;
	buildOptions.motionOptions.timeBegin		= 0.0f;
	buildOptions.motionOptions.timeEnd			= 0.0f;
	buildOptions.motionOptions.flags			= OPTIX_MOTION_FLAG_NONE;

	AccelStructBase::build(stream, allocator, optixBuildInputs, buildOptions);
}

/*********************************************************************************
***************************    InstAccelStructImpl    ****************************
*********************************************************************************/

namespace kernels
{
	__global__ void AssignInstanceTransforms(dev::Ptr<OptixInstance> pInstances, dev::Ptr<const dev::Ptr<const Mat4x4>> ppTransforms, unsigned int numInstances)
	{
		CUDA_for(i, numInstances);

		Mat4x4 transform = {};
		transform.rows[0] = ns::float4{ 1, 0, 0, 0 };
		transform.rows[1] = ns::float4{ 0, 1, 0, 0 };
		transform.rows[2] = ns::float4{ 0, 0, 1, 0 };
		transform.rows[3] = ns::float4{ 0, 0, 0, 1 };

		if (ppTransforms[i])
		{
			transform = *ppTransforms[i];
		}

		ns::float4 * pAddressBegin = reinterpret_cast<ns::float4*>(pInstances[i].transform);

		pAddressBegin[0] = transform.rows[0];
		pAddressBegin[1] = transform.rows[1];
		pAddressBegin[2] = transform.rows[2];
	}
}

void InstAccelStructImpl::build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<BuildInput> buildInputs, bool preferFastTrace, bool allowUpdate)
{
	m_buildInputs.resize(buildInputs.size());
	m_instances.resize(allocator, buildInputs.size());
	m_transforms.resize(allocator, buildInputs.size());

	std::vector<OptixInstance>					instances(buildInputs.size());
	std::vector<ns::dev::Ptr<const Mat4x4>>		pTransforms(buildInputs.size());

	for (size_t i = 0; i < buildInputs.size(); i++)
	{
		instances[i]						= OptixInstance{};
		instances[i].traversableHandle		= buildInputs[i].geomAccelStruct->handle();
		instances[i].visibilityMask			= buildInputs[i].visibilityMask;
		instances[i].instanceId				= buildInputs[i].instanceId;
		instances[i].sbtOffset				= buildInputs[i].sbtOffset;
		instances[i].flags					= buildInputs[i].flags;
		pTransforms[i]						= buildInputs[i].transform;
		m_buildInputs[i]					= buildInputs[i];
	}

	stream.memcpy(m_instances.data(), instances.data(), instances.size());
	stream.memcpy(m_transforms.data(), pTransforms.data(), pTransforms.size());
	stream.launch(kernels::AssignInstanceTransforms, ns::ceil_div(m_instances.size(), 128), 128)(m_instances, m_transforms, static_cast<uint32_t>(m_instances.size()));

	OptixBuildInput										optixBuildInput = {};
	optixBuildInput.type								= OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	optixBuildInput.instanceArray.instances				= (CUdeviceptr)m_instances.data();
#if OPTIX_VERSION >= 70600
	optixBuildInput.instanceArray.instanceStride		= sizeof(OptixInstance);
#endif
	optixBuildInput.instanceArray.numInstances			= static_cast<uint32_t>(m_instances.size());

	OptixAccelBuildOptions								buildOptions = {};
	buildOptions.operation								= OPTIX_BUILD_OPERATION_BUILD;
	buildOptions.buildFlags								= OPTIX_BUILD_FLAG_NONE;
//	buildOptions.buildFlags								|= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	buildOptions.buildFlags								|= preferFastTrace ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE : OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
	buildOptions.buildFlags								|= allowUpdate ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0;
	buildOptions.motionOptions.numKeys					= 0;
	buildOptions.motionOptions.timeBegin				= 0.0f;
	buildOptions.motionOptions.timeEnd					= 0.0f;
	buildOptions.motionOptions.flags					= OPTIX_MOTION_FLAG_NONE;

	AccelStructBase::build(stream, allocator, { optixBuildInput }, buildOptions);
}


void InstAccelStructImpl::rebuild(ns::Stream & stream)
{
	stream.launch(kernels::AssignInstanceTransforms, ns::ceil_div(m_instances.size(), 128), 128)(m_instances, m_transforms, static_cast<uint32_t>(m_instances.size()));

	AccelStructBase::rebuild(stream);
}


void InstAccelStructImpl::refit(ns::Stream & stream)
{
	stream.launch(kernels::AssignInstanceTransforms, ns::ceil_div(m_instances.size(), 128), 128)(m_instances, m_transforms, static_cast<uint32_t>(m_instances.size()));

	AccelStructBase::refit(stream);
}