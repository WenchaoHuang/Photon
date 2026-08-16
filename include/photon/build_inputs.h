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

#include "fwd.h"
#include <optix.h>
#include <type_traits>
#include <nucleus/span.h>
#include <nucleus/device_span.h>
#include <nucleus/vector_traits.h>

namespace PHOTON_NAMESPACE
{
	/*****************************************************************************
	*******************************    Concepts    *******************************
	*****************************************************************************/

	/**
	 *	@brief		Concept that matches raw integer types suitable for SBT index buffers.
	 *	@note		OptiX supports 8-bit, 16-bit, and 32-bit unsigned integer index buffers.
	 */
	template<typename Type> concept IndexType = std::unsigned_integral<Type> && (sizeof(Type) <= 4);


	/**
	 *	@brief		Concept that matches types suitable for triangle index buffers.
	 *	@note		OptiX supports 8-bit, 16-bit, and 32-bit unsigned integer index buffers, as well as three-component vectors of these types.
	 */
	template<typename Type> concept TripletIndexType = ns::uchar3_like<Type> || ns::ushort3_like<Type> || ns::uint3_like<Type>;

	/*****************************************************************************
	****************************    IndicesFormat    *****************************
	*****************************************************************************/

	//!	@brief		Enumeration for the format of triangle vertices in OptiX build inputs.
	enum class VertexFormat : unsigned int
	{
		Float2 = OPTIX_VERTEX_FORMAT_FLOAT2,
		Float3 = OPTIX_VERTEX_FORMAT_FLOAT3,
	};


	//!	@brief		Enumeration for the format of triangle indices in OptiX build inputs.
	enum class IndicesFormat : unsigned int
	{
		Uint3 = OPTIX_INDICES_FORMAT_UNSIGNED_INT3,
		Uchar3 = OPTIX_INDICES_FORMAT_UNSIGNED_BYTE3,
		Ushort3 = OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3,
	};

	/*****************************************************************************
	************************    BuildInputTriangles    ***************************
	*****************************************************************************/

	struct BuildInputTriangles : public OptixBuildInputTriangleArray
	{
		/**
		 *	@brief		Default constructor for the triangle build input, initializes members to default values.
		 */
		BuildInputTriangles() : OptixBuildInputTriangleArray{}
		{
			OptixBuildInputTriangleArray::numSbtRecords = 1;
		}

	public:

		/**
		 *	@brief		Set the vertex buffers for the triangle build input.
		 *	@param[in]	vertexBuffers - A span of device pointers representing the vertex buffers.
		 *	@param[in]	vertexFormat - The format of the triangle vertices in the buffers.
		 *	@param[in]	numVertices - The number of vertices in the buffers.
		 *	@param[in]	vertexStrideInBytes - The stride between consecutive vertices in bytes. Default is 0 (tightly packed).
		 */
		BuildInputTriangles & setVertexBuffers(ns::Span<const CUdeviceptr> vertexBuffers, VertexFormat vertexFormat, size_t numVertices, size_t vertexStrideInBytes = 0)
		{
			OptixBuildInputTriangleArray::vertexStrideInBytes = static_cast<unsigned int>(vertexStrideInBytes);
			OptixBuildInputTriangleArray::vertexFormat = static_cast<OptixVertexFormat>(vertexFormat);
			OptixBuildInputTriangleArray::numVertices = static_cast<unsigned int>(numVertices);
			OptixBuildInputTriangleArray::vertexBuffers = vertexBuffers.data();
			return *this;
		}

	public:

		/**
		 *	@brief		Set the index buffer for the triangle build input.
		 *	@param[in]	indexFormat - The format of the triangle indices in the buffer.
		 *	@param[in]	indexBuffer - A span of bytes representing the index buffer.
		 *	@param[in]	numTriplets - The number of triangle index triplets in the buffer.
		 *	@param[in]	strideInBytes - The stride between consecutive index triplets in bytes. Default is 0 (tightly packed).
		 */
		BuildInputTriangles & setIndexBuffer(dev::Span<const ns::byte> indexBuffer, IndicesFormat indexFormat, unsigned int strideInBytes = 0)
		{
			unsigned int indexSizeInBytes = (indexFormat == IndicesFormat::Uchar3) ? 3 : (indexFormat == IndicesFormat::Ushort3) ? 6 : 12;
			OptixBuildInputTriangleArray::numIndexTriplets = (strideInBytes == 0) ? static_cast<unsigned int>(indexBuffer.size() / indexSizeInBytes) : static_cast<unsigned int>(indexBuffer.size() / strideInBytes);
			OptixBuildInputTriangleArray::indexBuffer = reinterpret_cast<CUdeviceptr>(indexBuffer.data());
			OptixBuildInputTriangleArray::indexFormat = static_cast<OptixIndicesFormat>(indexFormat);
			OptixBuildInputTriangleArray::indexStrideInBytes = strideInBytes;

			return *this;
		}


		/**
		 *	@brief		Set the index buffer for the triangle build input using a span of raw integer types or three-component vector types.
		 *	@param[in]	indices - A span of raw integer types (8, 16, or 32-bit) or three-component vector types representing the triangle indices.
		 */
		template<typename Type> BuildInputTriangles & setIndexBuffer(dev::Span<const Type> indices) requires (IndexType<Type> || TripletIndexType<Type>)
		{
			OptixBuildInputTriangleArray::indexBuffer = reinterpret_cast<CUdeviceptr>(indices.data());

			if constexpr (IndexType<Type>)
			{
				OptixBuildInputTriangleArray::numIndexTriplets = static_cast<unsigned int>(indices.size() / 3);
				OptixBuildInputTriangleArray::indexStrideInBytes = 0;

				if constexpr (sizeof(Type) == 1)
					OptixBuildInputTriangleArray::indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_BYTE3;
				else if constexpr (sizeof(Type) == 2)
					OptixBuildInputTriangleArray::indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3;
				else if constexpr (sizeof(Type) == 4)
					OptixBuildInputTriangleArray::indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
			}
			else
			{
				OptixBuildInputTriangleArray::numIndexTriplets = static_cast<unsigned int>(indices.size());
				OptixBuildInputTriangleArray::indexStrideInBytes = sizeof(Type);

				if constexpr (ns::uchar3_like<Type>)
					OptixBuildInputTriangleArray::indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_BYTE3;
				else if constexpr (ns::ushort3_like<Type>)
					OptixBuildInputTriangleArray::indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3;
				else if constexpr (ns::uint3_like<Type>)
					OptixBuildInputTriangleArray::indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
			}
			return *this;
		}

	public:

		/**
		 *	@brief		Set the SBT index offsets for the triangle build input.
		 *	@param[in]	offsetBuffer - A span of bytes representing the SBT index offsets.
		 *	@param[in]	sizeInBytes - The size of the offset buffer in bytes, needs to be 1, 2 or 4 (8, 16 or 32 bit).
		 *	@param[in]	strideInBytes - The stride between consecutive offsets in bytes. Default is 0 (tightly packed).
		 */
		BuildInputTriangles & setSbtIndexOffsets(dev::Span<const ns::byte> offsetBuffer, unsigned int sizeInBytes, unsigned int strideInBytes = 0)
		{
			OptixBuildInputTriangleArray::numSbtRecords = (strideInBytes == 0) ? static_cast<unsigned int>(offsetBuffer.size() / sizeInBytes) : static_cast<unsigned int>(offsetBuffer.size() / strideInBytes);
			OptixBuildInputTriangleArray::sbtIndexOffsetBuffer = reinterpret_cast<CUdeviceptr>(offsetBuffer.data());
			OptixBuildInputTriangleArray::sbtIndexOffsetStrideInBytes = strideInBytes;
			OptixBuildInputTriangleArray::sbtIndexOffsetSizeInBytes = sizeInBytes;

			return *this;
		}


		/**
		 *	@brief		Set the SBT index offsets for the triangle build input.
		 *	@param[in]	offsets - A span of raw integer types representing the SBT index offsets.
		 */
		template<IndexType Type> BuildInputTriangles & setSbtIndexOffsets(dev::Span<const Type> offsets)
		{
			OptixBuildInputTriangleArray::numSbtRecords = static_cast<unsigned int>(offsets.size());
			OptixBuildInputTriangleArray::sbtIndexOffsetBuffer = reinterpret_cast<CUdeviceptr>(offsets.data());
			OptixBuildInputTriangleArray::sbtIndexOffsetSizeInBytes = sizeof(Type);
			OptixBuildInputTriangleArray::sbtIndexOffsetStrideInBytes = 0;

			return *this;
		}

	public:

		/**
		 *	@brief		Set the primitive index offset for the triangle build input.
		 *	@param[in]	indexOffset - Primitive index bias, applied in `optixGetPrimitiveIndex()`.
		 *	@warning	Sum of `primitiveIndexOffset` and number of triangles must not overflow 32bits.
		 */
		BuildInputTriangles & setPrimitiveIndexOffset(unsigned int indexOffset)
		{
			OptixBuildInputTriangleArray::primitiveIndexOffset = indexOffset;

			return *this;
		}
	};

	/*****************************************************************************
	***************************    BuildInputCurves    ***************************
	*****************************************************************************/

#if OPTIX_VERSION >= 70100
	struct BuildInputCurves : public OptixBuildInputCurveArray
	{
		/**
		 *	@brief		Default constructor for the curve build input, initializes members to default values.
		 */
		BuildInputCurves() : OptixBuildInputCurveArray{} {}

	public:

		/**
		 *	@brief		Set the primitive type for the curve build input.
		 *	@param[in]	type - The degree and basis of the curve primitives.
		 */
		BuildInputCurves & setCurveType(OptixPrimitiveType type)
		{
			OptixBuildInputCurveArray::curveType = type;

			return *this;
		}

	public:

		/**
		 *	@brief		Set the vertex buffers for the curve build input.
		 *	@param[in]	vertexBuffers - A span of device pointers representing the curve vertex buffers.
		 *	@param[in]	numVertices - The number of curve vertices in each buffer.
		 *	@param[in]	vertexStrideInBytes - The stride between consecutive curve vertices in bytes. Default is 0 (tightly packed).
		 */
		BuildInputCurves & setVertexBuffers(ns::Span<const CUdeviceptr> vertexBuffers, size_t numVertices, size_t vertexStrideInBytes = 0)
		{
			OptixBuildInputCurveArray::vertexStrideInBytes = static_cast<unsigned int>(vertexStrideInBytes);
			OptixBuildInputCurveArray::numVertices = static_cast<unsigned int>(numVertices);
			OptixBuildInputCurveArray::vertexBuffers = vertexBuffers.data();

			return *this;
		}


		/**
		 *	@brief		Set the width buffers for the curve build input.
		 *	@param[in]	widthBuffers - A span of device pointers representing the curve width buffers.
		 *	@param[in]	widthStrideInBytes - The stride between consecutive curve widths in bytes. Default is 0 (tightly packed).
		 */
		BuildInputCurves & setWidthBuffers(ns::Span<const CUdeviceptr> widthBuffers, size_t widthStrideInBytes = 0)
		{
			OptixBuildInputCurveArray::widthStrideInBytes = static_cast<unsigned int>(widthStrideInBytes);
			OptixBuildInputCurveArray::widthBuffers = widthBuffers.data();

			return *this;
		}

	public:

		/**
		 *	@brief		Set the index buffer for the curve build input.
		 *	@param[in]	indexBuffer - A span of bytes representing the curve segment indices.
		 *	@param[in]	strideInBytes - The stride between consecutive indices in bytes. Default is 0 (tightly packed).
		 */
		BuildInputCurves & setIndexBuffer(dev::Span<const ns::byte> indexBuffer, unsigned int strideInBytes = 0)
		{
			OptixBuildInputCurveArray::numPrimitives = (strideInBytes == 0) ? static_cast<unsigned int>(indexBuffer.size() / sizeof(unsigned int)) : static_cast<unsigned int>(indexBuffer.size() / strideInBytes);
			OptixBuildInputCurveArray::indexBuffer = reinterpret_cast<CUdeviceptr>(indexBuffer.data());
			OptixBuildInputCurveArray::indexStrideInBytes = strideInBytes;

			return *this;
		}


		/**
		 *	@brief		Set the index buffer for the curve build input.
		 *	@param[in]	indices - A span of unsigned integers representing the curve segment indices.
		 */
		BuildInputCurves & setIndexBuffer(dev::Span<const unsigned int> indices)
		{
			OptixBuildInputCurveArray::numPrimitives = static_cast<unsigned int>(indices.size());
			OptixBuildInputCurveArray::indexBuffer = reinterpret_cast<CUdeviceptr>(indices.data());
			OptixBuildInputCurveArray::indexStrideInBytes = 0;

			return *this;
		}

	public:

		/**
		 *	@brief		Set the geometry flags for the curve build input.
		 *	@param[in]	flags - A combination of `OptixGeometryFlags` describing the primitive behavior.
		 */
		BuildInputCurves & setGeometryFlags(unsigned int flags)
		{
			OptixBuildInputCurveArray::flag = flags;

			return *this;
		}


		/**
		 *	@brief		Set the end cap flags for the curve build input.
		 *	@param[in]	flags - A combination of `OptixCurveEndcapFlags` describing the curve end caps.
		 */
		BuildInputCurves & setEndcapFlags(unsigned int flags)
		{
			OptixBuildInputCurveArray::endcapFlags = flags;

			return *this;
		}

	public:

		/**
		 *	@brief		Set the primitive index offset for the curve build input.
		 *	@param[in]	indexOffset - Primitive index bias, applied in `optixGetPrimitiveIndex()`.
		 *	@warning	Sum of `primitiveIndexOffset` and number of curve segments must not overflow 32bits.
		 */
		BuildInputCurves & setPrimitiveIndexOffset(unsigned int indexOffset)
		{
			OptixBuildInputCurveArray::primitiveIndexOffset = indexOffset;

			return *this;
		}
	};
#endif

	/*****************************************************************************
	**************************    BuildInputSpheres    ***************************
	*****************************************************************************/

	struct BuildInputSpheres : public OptixBuildInputSphereArray
	{
		/**
		 *	@brief		Default constructor for the sphere build input, initializes members to default values.
		 */
		BuildInputSpheres() : OptixBuildInputSphereArray{}
		{
			OptixBuildInputSphereArray::numSbtRecords = 1;
		}

	public:

		/**
		 *	@brief		Set the vertex buffers for the sphere build input.
		 *	@param[in]	vertexBuffers - A span of device pointers representing the sphere center buffers.
		 *	@param[in]	numVertices - The number of sphere centers in each buffer.
		 *	@param[in]	vertexStrideInBytes - The stride between consecutive sphere centers in bytes. Default is 0 (tightly packed).
		 */
		BuildInputSpheres & setVertexBuffers(ns::Span<const CUdeviceptr> vertexBuffers, size_t numVertices, size_t vertexStrideInBytes = 0)
		{
			OptixBuildInputSphereArray::vertexStrideInBytes = static_cast<unsigned int>(vertexStrideInBytes);
			OptixBuildInputSphereArray::numVertices = static_cast<unsigned int>(numVertices);
			OptixBuildInputSphereArray::vertexBuffers = vertexBuffers.data();

			return *this;
		}


		/**
		 *	@brief		Set the radius buffers for the sphere build input.
		 *	@param[in]	radiusBuffers - A span of device pointers representing the sphere radius buffers.
		 *	@param[in]	singleRadius - Whether each radius buffer contains a single radius shared by all spheres.
		 *	@param[in]	radiusStrideInBytes - The stride between consecutive radii in bytes. Default is 0 (tightly packed).
		 */
		BuildInputSpheres & setRadiusBuffers(ns::Span<const CUdeviceptr> radiusBuffers, bool singleRadius = false, size_t radiusStrideInBytes = 0)
		{
			OptixBuildInputSphereArray::radiusStrideInBytes = static_cast<unsigned int>(radiusStrideInBytes);
			OptixBuildInputSphereArray::singleRadius = static_cast<int>(singleRadius);
			OptixBuildInputSphereArray::radiusBuffers = radiusBuffers.data();

			return *this;
		}

	public:

		/**
		 *	@brief		Set the SBT index offsets for the sphere build input.
		 *	@param[in]	offsetBuffer - A span of bytes representing the SBT index offsets.
		 *	@param[in]	sizeInBytes - The size of the offset buffer in bytes, needs to be 1, 2 or 4 (8, 16 or 32 bit).
		 *	@param[in]	strideInBytes - The stride between consecutive offsets in bytes. Default is 0 (tightly packed).
		 */
		BuildInputSpheres & setSbtIndexOffsets(dev::Span<const ns::byte> offsetBuffer, unsigned int sizeInBytes, unsigned int strideInBytes = 0)
		{
			OptixBuildInputSphereArray::numSbtRecords = (strideInBytes == 0) ? static_cast<unsigned int>(offsetBuffer.size() / sizeInBytes) : static_cast<unsigned int>(offsetBuffer.size() / strideInBytes);
			OptixBuildInputSphereArray::sbtIndexOffsetBuffer = reinterpret_cast<CUdeviceptr>(offsetBuffer.data());
			OptixBuildInputSphereArray::sbtIndexOffsetStrideInBytes = strideInBytes;
			OptixBuildInputSphereArray::sbtIndexOffsetSizeInBytes = sizeInBytes;

			return *this;
		}


		/**
		 *	@brief		Set the SBT index offsets for the sphere build input.
		 *	@param[in]	offsets - A span of raw integer types representing the SBT index offsets.
		 */
		template<IndexType Type> BuildInputSpheres & setSbtIndexOffsets(dev::Span<const Type> offsets)
		{
			OptixBuildInputSphereArray::numSbtRecords = static_cast<unsigned int>(offsets.size());
			OptixBuildInputSphereArray::sbtIndexOffsetBuffer = reinterpret_cast<CUdeviceptr>(offsets.data());
			OptixBuildInputSphereArray::sbtIndexOffsetSizeInBytes = sizeof(Type);
			OptixBuildInputSphereArray::sbtIndexOffsetStrideInBytes = 0;

			return *this;
		}

	public:

		/**
		 *	@brief		Set the primitive index offset for the sphere build input.
		 *	@param[in]	indexOffset - Primitive index bias, applied in `optixGetPrimitiveIndex()`.
		 *	@warning	Sum of `primitiveIndexOffset` and number of spheres must not overflow 32bits.
		 */
		BuildInputSpheres & setPrimitiveIndexOffset(unsigned int indexOffset)
		{
			OptixBuildInputSphereArray::primitiveIndexOffset = indexOffset;

			return *this;
		}
	};

	/*****************************************************************************
	***************************    BuildInputAabbs    ****************************
	*****************************************************************************/

	struct BuildInputAabbs : public OptixBuildInputCustomPrimitiveArray
	{
		/**
		 *	@brief		Default constructor for the AABB build input, initializes members to default values.
		 */
		BuildInputAabbs() : OptixBuildInputCustomPrimitiveArray{}
		{
			OptixBuildInputCustomPrimitiveArray::numSbtRecords = 1;
		}

	public:

		/**
		 *	@brief		Set the AABB buffers for the custom primitive build input.
		 *	@param[in]	aabbBuffers - A span of device pointers representing the AABB buffers.
		 *	@param[in]	numPrimitives - The number of AABBs in each buffer.
		 *	@param[in]	strideInBytes - The stride between consecutive AABBs in bytes. Default is 0 (tightly packed).
		 */
		BuildInputAabbs & setAabbBuffers(ns::Span<const CUdeviceptr> aabbBuffers, size_t numPrimitives, size_t strideInBytes = 0)
		{
			OptixBuildInputCustomPrimitiveArray::numPrimitives = static_cast<unsigned int>(numPrimitives);
			OptixBuildInputCustomPrimitiveArray::strideInBytes = static_cast<unsigned int>(strideInBytes);
			OptixBuildInputCustomPrimitiveArray::aabbBuffers = aabbBuffers.data();

			return *this;
		}

	public:

		/**
		 *	@brief		Set the SBT index offsets for the custom primitive build input.
		 *	@param[in]	offsetBuffer - A span of bytes representing the SBT index offsets.
		 *	@param[in]	sizeInBytes - The size of the offset buffer in bytes, needs to be 1, 2 or 4 (8, 16 or 32 bit).
		 *	@param[in]	strideInBytes - The stride between consecutive offsets in bytes. Default is 0 (tightly packed).
		 */
		BuildInputAabbs & setSbtIndexOffsets(dev::Span<const ns::byte> offsetBuffer, unsigned int sizeInBytes, unsigned int strideInBytes = 0)
		{
			OptixBuildInputCustomPrimitiveArray::numSbtRecords = (strideInBytes == 0) ? static_cast<unsigned int>(offsetBuffer.size() / sizeInBytes) : static_cast<unsigned int>(offsetBuffer.size() / strideInBytes);
			OptixBuildInputCustomPrimitiveArray::sbtIndexOffsetBuffer = reinterpret_cast<CUdeviceptr>(offsetBuffer.data());
			OptixBuildInputCustomPrimitiveArray::sbtIndexOffsetStrideInBytes = strideInBytes;
			OptixBuildInputCustomPrimitiveArray::sbtIndexOffsetSizeInBytes = sizeInBytes;

			return *this;
		}


		/**
		 *	@brief		Set the SBT index offsets for the custom primitive build input.
		 *	@param[in]	offsets - A span of raw integer types representing the SBT index offsets.
		 */
		template<IndexType Type> BuildInputAabbs & setSbtIndexOffsets(dev::Span<const Type> offsets)
		{
			OptixBuildInputCustomPrimitiveArray::numSbtRecords = static_cast<unsigned int>(offsets.size());
			OptixBuildInputCustomPrimitiveArray::sbtIndexOffsetBuffer = reinterpret_cast<CUdeviceptr>(offsets.data());
			OptixBuildInputCustomPrimitiveArray::sbtIndexOffsetSizeInBytes = sizeof(Type);
			OptixBuildInputCustomPrimitiveArray::sbtIndexOffsetStrideInBytes = 0;

			return *this;
		}

	public:

		/**
		 *	@brief		Set the primitive index offset for the custom primitive build input.
		 *	@param[in]	indexOffset - Primitive index bias, applied in `optixGetPrimitiveIndex()`.
		 *	@warning	Sum of `primitiveIndexOffset` and number of AABBs must not overflow 32bits.
		 */
		BuildInputAabbs & setPrimitiveIndexOffset(unsigned int indexOffset)
		{
			OptixBuildInputCustomPrimitiveArray::primitiveIndexOffset = indexOffset;

			return *this;
		}
	};
}
