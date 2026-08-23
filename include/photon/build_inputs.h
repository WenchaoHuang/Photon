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
#include <vector>
#include <optix.h>
#include <concepts>
#include <algorithm>
#include <type_traits>
#include <nucleus/span.h>
#include <nucleus/device_span.h>
#include <nucleus/vector_traits.h>

namespace PHOTON_NAMESPACE
{
	/*****************************************************************************
	*******************************    Concepts    *******************************
	*****************************************************************************/

	//!	@brief		Matches the floating-point vector layouts supported by OptiX triangle vertices.
	template<typename Type> concept VertexType = ns::float2_like<Type> || ns::float3_like<Type>;


	//!	@brief		Matches floating-point curve control-point layouts, including padded float4 points.
	template<typename Type> concept CurveVertexType = ns::float3_like<Type> || ns::float4_like<Type>;


	/**
	 *	@brief		Concept that matches raw integer types suitable for SBT index buffers.
	 *	@note		OptiX supports 8-bit, 16-bit, and 32-bit singed and unsigned integer index buffers.
	 */
	template<typename Type> concept IndexType =
		std::is_same_v<Type, char>			|| std::is_same_v<Type, short>			|| std::is_same_v<Type, int> ||
		std::is_same_v<Type, unsigned char> || std::is_same_v<Type, unsigned short> || std::is_same_v<Type, unsigned int>;


	/**
	 *	@brief		Concept that matches types suitable for triangle index buffers.
	 *	@note		OptiX supports 8-bit, 16-bit, and 32-bit signed and unsigned integer index buffers,
	 *				as well as three-component vectors of these types.
	 */
	template<typename Type> concept TriangleIndexType = IndexType<Type> ||
		ns::char3_like<Type>  || ns::short3_like<Type>  || ns::int3_like<Type> || 
		ns::uchar3_like<Type> || ns::ushort3_like<Type> || ns::uint3_like<Type>;

	/*****************************************************************************
	****************************    IndicesFormat    *****************************
	*****************************************************************************/

	//!	@brief		Enumeration for the format of triangle vertices in OptiX build inputs.
	enum class VertexFormat : unsigned int
	{
		Float2 = OPTIX_VERTEX_FORMAT_FLOAT2,
		Float3 = OPTIX_VERTEX_FORMAT_FLOAT3,
	};


	//!	
	template<VertexType Type> struct VertexFormatOf;
	template<ns::float2_like Type> struct VertexFormatOf<Type> { static constexpr VertexFormat value = VertexFormat::Float2; };
	template<ns::float3_like Type> struct VertexFormatOf<Type> { static constexpr VertexFormat value = VertexFormat::Float3; };


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

	/**
	 *	@brief		Owning wrapper for `OptixBuildInputTriangleArray`.
	 *	@details	OptiX descriptors contain host pointers to arrays of device addresses and
	 *				geometry flags. This wrapper owns those host arrays with `std::vector`, so
	 *				copying or moving the wrapper cannot leave the descriptor bound to another
	 *				object's storage. Typed setters infer layout metadata, while raw setters keep
	 *				explicit control for interleaved or otherwise custom buffer layouts.
	 */
	class BuildInputTriangles
	{

	public:

		/**
		 *	@brief		Set the vertex buffers for the triangle build input.
		 *	@param[in]	vertexBuffers - A span of device byte spans, one for each motion step.
		 *	@param[in]	vertexFormat - The format of the triangle vertices in the buffers.
		 *	@param[in]	numVertices - The number of vertices in the buffers.
		 *	@param[in]	vertexStrideInBytes - The stride between consecutive vertices in bytes. Default is 0 (tightly packed).
		 */
		BuildInputTriangles & setVertexBuffers(ns::Span<const dev::Span<const ns::byte>> vertexBuffers, VertexFormat vertexFormat, size_t numVertices, size_t vertexStrideInBytes = 0)
		{
			m_vertexBuffers.resize(vertexBuffers.size());
			for (size_t i = 0; i < vertexBuffers.size(); i++)
				m_vertexBuffers[i] = reinterpret_cast<CUdeviceptr>(vertexBuffers[i].data());
			m_native.vertexStrideInBytes = static_cast<unsigned int>(vertexStrideInBytes);
			m_native.vertexFormat = static_cast<OptixVertexFormat>(vertexFormat);
			m_native.numVertices = static_cast<unsigned int>(numVertices);
			return *this;
		}


		/**
		 *	@brief		Set typed vertex buffers and infer format, vertex count and stride.
		 *	@param[in]	vertexBuffers - One device span per motion step.
		 *	@note		Every motion step must contain the same number of vertices.
		 *				The inferred stride is `sizeof(Type)` and the format follows the matched vector type.
		 */
		template<VertexType Type> BuildInputTriangles & setVertexBuffers(ns::Span<const dev::Span<const Type>> vertexBuffers)
		{
			for (const auto & vertexBuffer : vertexBuffers)
				NS_ASSERT(vertexBuffers.empty() || vertexBuffer.size() == vertexBuffers.front().size());

			m_vertexBuffers.resize(vertexBuffers.size());
			for (size_t i = 0; i < vertexBuffers.size(); ++i)
				m_vertexBuffers[i] = reinterpret_cast<CUdeviceptr>(vertexBuffers[i].data());
			m_native.numVertices = vertexBuffers.empty() ? 0u : static_cast<unsigned int>(vertexBuffers.front().size());
			m_native.vertexFormat = static_cast<OptixVertexFormat>(VertexFormatOf<Type>::value);
			m_native.vertexStrideInBytes = static_cast<unsigned int>(sizeof(Type));
			return *this;
		}


		/**
		 *	@brief		Set one typed vertex buffer and infer its format, count and stride.
		 *	@param[in]	vertexBuffer - Typed device storage for a single motion step.
		 */
		template<VertexType Type> BuildInputTriangles & setVertexBuffer(dev::Span<const Type> vertexBuffer)
		{
			return setVertexBuffers<Type>({ vertexBuffer });
		}

	public:

		/**
		 *	@brief		Set the index buffer for the triangle build input.
		 *	@param[in]	indexFormat - The format of the triangle indices in the buffer.
		 *	@param[in]	indexBuffer - A span of bytes representing the index buffer.
		 *	@param[in]	strideInBytes - The stride between consecutive index triplets in bytes. Default is 0 (tightly packed).
		 */
		BuildInputTriangles & setIndexBuffer(dev::Span<const ns::byte> indexBuffer, IndicesFormat indexFormat, unsigned int strideInBytes = 0)
		{
			unsigned int indexSizeInBytes = (indexFormat == IndicesFormat::Uchar3) ? 3 : (indexFormat == IndicesFormat::Ushort3) ? 6 : 12;

			NS_ASSERT(strideInBytes == 0 || strideInBytes >= indexSizeInBytes);
			m_native.numIndexTriplets = (strideInBytes == 0) ? static_cast<unsigned int>(indexBuffer.size() / indexSizeInBytes) : static_cast<unsigned int>(indexBuffer.size() / strideInBytes);
			m_native.indexBuffer = reinterpret_cast<CUdeviceptr>(indexBuffer.data());
			m_native.indexFormat = static_cast<OptixIndicesFormat>(indexFormat);
			m_native.indexStrideInBytes = strideInBytes;
			return *this;
		}


		/**
		 *	@brief		Set the index buffer for the triangle build input using a span of raw integer types or three-component vector types.
		 *	@param[in]	indices - A span of raw integer types (8, 16, or 32-bit) or three-component vector types representing the triangle indices.
		 */
		template<TriangleIndexType Type> BuildInputTriangles & setIndexBuffer(dev::Span<const Type> indices)
		{
			m_native.indexBuffer = reinterpret_cast<CUdeviceptr>(indices.data());

			if constexpr (IndexType<Type>)
			{
				NS_ASSERT(indices.size() % 3 == 0);
				m_native.numIndexTriplets = static_cast<unsigned int>(indices.size() / 3);
				m_native.indexStrideInBytes = static_cast<unsigned int>(3 * sizeof(Type));

				if constexpr (sizeof(Type) == 1)
					m_native.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_BYTE3;
				else if constexpr (sizeof(Type) == 2)
					m_native.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3;
				else if constexpr (sizeof(Type) == 4)
					m_native.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
			}
			else
			{
				m_native.indexStrideInBytes = sizeof(Type);
				m_native.numIndexTriplets = static_cast<unsigned int>(indices.size());

				if constexpr (ns::char3_like<Type> || ns::uchar3_like<Type>)
					m_native.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_BYTE3;
				else if constexpr (ns::short3_like<Type> || ns::ushort3_like<Type>)
					m_native.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3;
				else if constexpr (ns::int3_like<Type> || ns::ushort3_like<Type>)
					m_native.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
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
			NS_ASSERT(strideInBytes == 0 || strideInBytes >= sizeInBytes);
			NS_ASSERT(sizeInBytes == 1 || sizeInBytes == 2 || sizeInBytes == 4);
			m_native.sbtIndexOffsetBuffer = reinterpret_cast<CUdeviceptr>(offsetBuffer.data());
			m_native.sbtIndexOffsetStrideInBytes = strideInBytes;
			m_native.sbtIndexOffsetSizeInBytes = sizeInBytes;
			return *this;
		}


		/**
		 *	@brief		Set typed SBT index offsets and infer element size and stride.
		 *	@param[in]	offsets - A span of raw integer types representing the SBT index offsets.
		 *	@note		This function deliberately does not change `numSbtRecords`; geometry flags
		 *				and the SBT record count are configured together by `setGeometryFlags()`.
		 */
		template<IndexType Type> BuildInputTriangles & setSbtIndexOffsets(dev::Span<const Type> offsets)
		{
			m_native.sbtIndexOffsetBuffer = reinterpret_cast<CUdeviceptr>(offsets.data());
			m_native.sbtIndexOffsetStrideInBytes = sizeof(Type);
			m_native.sbtIndexOffsetSizeInBytes = sizeof(Type);
			return *this;
		}

	public:

		/**
		 *	@brief		Set per-SBT geometry flags and synchronize `numSbtRecords`.
		 *	@param[in]	flags - One geometry flag for every SBT record.
		 */
		BuildInputTriangles & setGeometryFlags(ns::Span<const OptixGeometryFlags> flags)
		{
			NS_ASSERT(!flags.empty());
			m_geometryFlags.resize(flags.size());
			std::transform(flags.begin(), flags.end(), m_geometryFlags.begin(), [](OptixGeometryFlags flag) { return static_cast<unsigned int>(flag); });
			m_native.numSbtRecords = static_cast<unsigned int>(flags.size());
			return *this;
		}


		/**
		 *	@brief		Set the SBT record count and replicate one geometry flag for every record.
		 *	@param[in]	flag - Geometry behavior assigned to every record.
		 *	@param[in]	numSbtRecords - Number of SBT records available to this build input.
		 */
		BuildInputTriangles & setGeometryFlags(OptixGeometryFlags flag, unsigned int numSbtRecords = 1)
		{
			NS_ASSERT(numSbtRecords > 0);
			m_geometryFlags.assign(numSbtRecords, static_cast<unsigned int>(flag));
			m_native.numSbtRecords = numSbtRecords;
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
			m_native.primitiveIndexOffset = indexOffset;
			return *this;
		}

	public:

		/**
		 *	@brief		Create a native OptiX descriptor bound to this object's host storage.
		 *	@note		The returned descriptor is a lightweight copy. Its host pointers remain
		 *				valid only while this wrapper remains alive and is not modified.
		 */
		[[nodiscard]] OptixBuildInputTriangleArray native() const noexcept
		{
			m_native.vertexBuffers = m_vertexBuffers.empty() ? nullptr : m_vertexBuffers.data();
			m_native.flags = m_geometryFlags.empty() ? nullptr : m_geometryFlags.data();
			return m_native;
		}

	private:

		std::vector<CUdeviceptr>					m_vertexBuffers;
		std::vector<unsigned int>					m_geometryFlags{ OPTIX_GEOMETRY_FLAG_NONE };
		mutable OptixBuildInputTriangleArray		m_native{ .numSbtRecords = 1 };
	};

	/*****************************************************************************
	***************************    BuildInputCurves    ***************************
	*****************************************************************************/

#if OPTIX_VERSION >= 70100
	/**
	 *	@brief		Owning wrapper for `OptixBuildInputCurveArray`.
	 *	@details	The host-side arrays of vertex and width device addresses are copied into
	 *				this object. Typed setters derive element strides and counts, while raw
	 *				setters preserve explicit control over custom layouts.
	 */
	class BuildInputCurves
	{

	public:

		/**
		 *	@brief		Set raw curve vertex buffers with explicit count and stride metadata.
		 *	@param[in]	vertexBuffers - One untyped device span for each motion step. Only the
		 *				device address is retained; the host-side address array is owned by this wrapper.
		 *	@param[in]	numVertices - Number of curve control points in every motion-step buffer.
		 *	@param[in]	vertexStrideInBytes - Explicit byte stride, or zero for tightly packed data.
		 */
		BuildInputCurves & setVertexBuffers(ns::Span<const dev::Span<const ns::byte>> vertexBuffers, size_t numVertices, size_t vertexStrideInBytes = 0)
		{
			m_vertexBuffers.resize(vertexBuffers.size());
			for (size_t i = 0; i < vertexBuffers.size(); ++i)
				m_vertexBuffers[i] = reinterpret_cast<CUdeviceptr>(vertexBuffers[i].data());
			m_native.vertexStrideInBytes = static_cast<unsigned int>(vertexStrideInBytes);
			m_native.numVertices = static_cast<unsigned int>(numVertices);
			return *this;
		}


		/**
		 *	@brief		Set typed curve vertices and infer the element stride and count.
		 *	@param[in]	vertexBuffers - One typed device span per motion step.
		 *	@note		Every motion step must contain the same number of control points.
		 */
		template<CurveVertexType Type> BuildInputCurves & setVertexBuffers(ns::Span<const dev::Span<const Type>> vertexBuffers)
		{
			for (const auto & vertexBuffer : vertexBuffers)
				NS_ASSERT(vertexBuffers.empty() || vertexBuffer.size() == vertexBuffers.front().size());

			m_vertexBuffers.resize(vertexBuffers.size());
			for (size_t i = 0; i < vertexBuffers.size(); ++i)
				m_vertexBuffers[i] = reinterpret_cast<CUdeviceptr>(vertexBuffers[i].data());
			m_native.numVertices = vertexBuffers.empty() ? 0u : static_cast<unsigned int>(vertexBuffers.front().size());
			m_native.vertexStrideInBytes = static_cast<unsigned int>(sizeof(Type));
			return *this;
		}


		/**
		 *	@brief		Set one typed curve vertex buffer and infer its count and stride.
		 *	@param[in]	vertexBuffer - Typed control-point storage for a single motion step.
		 */
		template<CurveVertexType Type> BuildInputCurves & setVertexBuffers(dev::Span<const Type> vertexBuffer)
		{
			m_vertexBuffers.assign(1, reinterpret_cast<CUdeviceptr>(vertexBuffer.data()));
			m_native.vertexStrideInBytes = static_cast<unsigned int>(sizeof(Type));
			m_native.numVertices = static_cast<unsigned int>(vertexBuffer.size());
			return *this;
		}

	public:

		/**
		 *	@brief		Set raw curve width buffers with an explicit element stride.
		 *	@param[in]	widthBuffers - One untyped device span for each motion step. Only the
		 *				device address is retained; the host-side address array is owned by this wrapper.
		 *	@param[in]	widthStrideInBytes - The stride between consecutive curve widths in bytes. Default is 0 (tightly packed).
		 */
		BuildInputCurves & setWidthBuffers(ns::Span<const dev::Span<const ns::byte>> widthBuffers, size_t widthStrideInBytes = 0)
		{
			m_widthBuffers.resize(widthBuffers.size());
			for (size_t i = 0; i < widthBuffers.size(); ++i)
				m_widthBuffers[i] = reinterpret_cast<CUdeviceptr>(widthBuffers[i].data());
			m_native.widthStrideInBytes = static_cast<unsigned int>(widthStrideInBytes);
			return *this;
		}


		/**
		 *	@brief		Set typed curve widths and infer the width stride.
		 *	@param[in]	widthBuffers - One floating-point device span per motion step.
		 */
		BuildInputCurves & setWidthBuffers(ns::Span<const dev::Span<const float>> widthBuffers)
		{
			m_widthBuffers.resize(widthBuffers.size());
			for (size_t i = 0; i < widthBuffers.size(); ++i)
				m_widthBuffers[i] = reinterpret_cast<CUdeviceptr>(widthBuffers[i].data());
			m_native.widthStrideInBytes = static_cast<unsigned int>(sizeof(float));
			return *this;
		}


		/**
		 *	@brief		Set one typed curve width buffer and infer its element stride.
		 *	@param[in]	widthBuffer - Floating-point width storage for a single motion step.
		 */
		BuildInputCurves & setWidthBuffer(dev::Span<const float> widthBuffer)
		{
			return setWidthBuffers({ widthBuffer });
		}

	public:

		/**
		 *	@brief		Set the index buffer for the curve build input.
		 *	@param[in]	indexBuffer - A span of bytes representing the curve segment indices.
		 *	@param[in]	strideInBytes - The stride between consecutive indices in bytes. Default is 0 (tightly packed).
		 */
		BuildInputCurves & setIndexBuffer(dev::Span<const ns::byte> indexBuffer, unsigned int strideInBytes = 0)
		{
			m_native.numPrimitives = (strideInBytes == 0) ? static_cast<unsigned int>(indexBuffer.size() / sizeof(unsigned int)) : static_cast<unsigned int>(indexBuffer.size() / strideInBytes);
			m_native.indexBuffer = reinterpret_cast<CUdeviceptr>(indexBuffer.data());
			m_native.indexStrideInBytes = strideInBytes;
			return *this;
		}


		/**
		 *	@brief		Set the index buffer for the curve build input.
		 *	@param[in]	indices - A span of unsigned integers representing the curve segment indices.
		 */
		template<IndexType Type> BuildInputCurves & setIndexBuffer(dev::Span<const Type> indices)
		{
			m_native.numPrimitives = static_cast<unsigned int>(indices.size());
			m_native.indexBuffer = reinterpret_cast<CUdeviceptr>(indices.data());
			m_native.indexStrideInBytes = sizeof(Type);
			return *this;
		}

	public:

		/**
		 *	@brief		Set the primitive type for the curve build input.
		 *	@param[in]	type - The degree and basis of the curve primitives.
		 */
		BuildInputCurves & setCurveType(OptixPrimitiveType type)
		{
			m_native.curveType = type;
			return *this;
		}


		/**
		 *	@brief		Set the geometry flags for the curve build input.
		 *	@param[in]	flags - A combination of `OptixGeometryFlags` describing the primitive behavior.
		 */
		BuildInputCurves & setGeometryFlags(unsigned int flags)
		{
			m_native.flag = flags;
			return *this;
		}


		/**
		 *	@brief		Set the end cap flags for the curve build input.
		 *	@param[in]	flags - A combination of `OptixCurveEndcapFlags` describing the curve end caps.
		 */
		BuildInputCurves & setEndcapFlags(unsigned int flags)
		{
			m_native.endcapFlags = flags;
			return *this;
		}


		/**
		 *	@brief		Set the primitive index offset for the curve build input.
		 *	@param[in]	indexOffset - Primitive index bias, applied in `optixGetPrimitiveIndex()`.
		 *	@warning	Sum of `primitiveIndexOffset` and number of curve segments must not overflow 32bits.
		 */
		BuildInputCurves & setPrimitiveIndexOffset(unsigned int indexOffset)
		{
			m_native.primitiveIndexOffset = indexOffset;
			return *this;
		}

	public:

		/**
		 *	@brief		Create a native curve descriptor bound to this wrapper's host arrays.
		 *	@note		The returned descriptor is a lightweight copy. Its host pointers remain
		 *				valid only while this wrapper remains alive and is not modified.
		 */
		[[nodiscard]] OptixBuildInputCurveArray native() const noexcept
		{
			NS_ASSERT(m_native.curveType != OPTIX_PRIMITIVE_TYPE_CUSTOM && "Curve primitive type must be specified.");
			m_native.vertexBuffers = m_vertexBuffers.empty() ? nullptr : m_vertexBuffers.data();
			m_native.widthBuffers = m_widthBuffers.empty() ? nullptr : m_widthBuffers.data();
			return m_native;
		}

	private:

		std::vector<CUdeviceptr>				m_vertexBuffers;
		std::vector<CUdeviceptr>				m_widthBuffers;
		mutable OptixBuildInputCurveArray		m_native{};
	};
#endif

	/*****************************************************************************
	**************************    BuildInputSpheres    ***************************
	*****************************************************************************/

	/**
	 *	@brief		Owning wrapper for `OptixBuildInputSphereArray`.
	 *	@details	The wrapper owns the host arrays referenced by the native descriptor and
	 *				rebinds them after copy or move operations. Typed setters infer sphere-center
	 *				and radius strides; raw setters support custom memory layouts.
	 */
	class BuildInputSpheres
	{

	public:

		/**
		 *	@brief		Set raw sphere-center buffers with explicit count and stride metadata.
		 *	@param[in]	vertexBuffers - One untyped device span for each motion step. Only the
		 *				device address is retained; the host-side address array is owned by this wrapper.
		 *	@param[in]	numVertices - The number of sphere centers in each buffer.
		 *	@param[in]	vertexStrideInBytes - The stride between consecutive sphere centers in bytes. Default is 0 (tightly packed).
		 */
		BuildInputSpheres & setVertexBuffers(ns::Span<const dev::Span<const ns::byte>> vertexBuffers, size_t numVertices, size_t vertexStrideInBytes = 0)
		{
			m_vertexBuffers.resize(vertexBuffers.size());
			for (size_t i = 0; i < vertexBuffers.size(); ++i)
				m_vertexBuffers[i] = reinterpret_cast<CUdeviceptr>(vertexBuffers[i].data());
			m_native.vertexStrideInBytes = static_cast<unsigned int>(vertexStrideInBytes);
			m_native.numVertices = static_cast<unsigned int>(numVertices);
			return *this;
		}


		/**
		 *	@brief		Set typed sphere centers and infer vertex count and stride.
		 *	@param[in]	vertexBuffers - One float3-like device span per motion step.
		 *	@note		Every motion step must contain the same number of sphere centers.
		 */
		template<ns::float3_like Type> BuildInputSpheres & setVertexBuffers(ns::Span<const dev::Span<const Type>> vertexBuffers)
		{
			for (const auto & vertexBuffer : vertexBuffers)
				NS_ASSERT(vertexBuffers.empty() || vertexBuffer.size() == vertexBuffers.front().size());

			m_vertexBuffers.resize(vertexBuffers.size());
			for (size_t i = 0; i < vertexBuffers.size(); ++i)
				m_vertexBuffers[i] = reinterpret_cast<CUdeviceptr>(vertexBuffers[i].data());
			m_native.numVertices = vertexBuffers.empty() ? 0u : static_cast<unsigned int>(vertexBuffers.front().size());
			m_native.vertexStrideInBytes = static_cast<unsigned int>(sizeof(Type));
			return *this;
		}


		/**
		 *	@brief		Set one typed sphere-center buffer and infer its count and stride.
		 *	@param[in]	vertexBuffer - Float3-like center storage for a single motion step.
		 */
		template<ns::float3_like Type> BuildInputSpheres & setVertexBuffer(dev::Span<const Type> vertexBuffer)
		{
			return setVertexBuffers<Type>({ vertexBuffer });
		}

	public:

		/**
		 *	@brief		Set raw sphere-radius buffers with an explicit element stride.
		 *	@param[in]	radiusBuffers - One untyped device span for each motion step. Only the
		 *				device address is retained; the host-side address array is owned by this wrapper.
		 *	@param[in]	singleRadius - Whether each radius buffer contains a single radius shared by all spheres.
		 *	@param[in]	radiusStrideInBytes - The stride between consecutive radii in bytes. Default is 0 (tightly packed).
		 */
		BuildInputSpheres & setRadiusBuffers(ns::Span<const dev::Span<const ns::byte>> radiusBuffers, bool singleRadius = false, size_t radiusStrideInBytes = 0)
		{
			m_radiusBuffers.resize(radiusBuffers.size());
			for (size_t i = 0; i < radiusBuffers.size(); ++i)
				m_radiusBuffers[i] = reinterpret_cast<CUdeviceptr>(radiusBuffers[i].data());
			m_native.radiusStrideInBytes = static_cast<unsigned int>(radiusStrideInBytes);
			m_native.singleRadius = static_cast<int>(singleRadius);
			return *this;
		}


		/**
		 *	@brief		Set typed sphere radii and infer the radius stride.
		 *	@param[in]	radiusBuffers - One floating-point device span per motion step.
		 *	@param[in]	singleRadius - Whether each motion step stores one shared radius.
		 */
		BuildInputSpheres & setRadiusBuffers(ns::Span<const dev::Span<const float>> radiusBuffers, bool singleRadius = false)
		{
			m_radiusBuffers.resize(radiusBuffers.size());
			for (size_t i = 0; i < radiusBuffers.size(); ++i)
				m_radiusBuffers[i] = reinterpret_cast<CUdeviceptr>(radiusBuffers[i].data());
			m_native.radiusStrideInBytes = static_cast<unsigned int>(sizeof(float));
			m_native.singleRadius = static_cast<int>(singleRadius);
			return *this;
		}


		/**
		 *	@brief		Set one typed sphere-radius buffer and infer its element stride.
		 *	@param[in]	radiusBuffer - Floating-point radius storage for a single motion step.
		 *	@param[in]	singleRadius - Whether the span stores one radius shared by all spheres.
		 */
		BuildInputSpheres & setRadiusBuffer(dev::Span<const float> radiusBuffer, bool singleRadius = false)
		{
			return setRadiusBuffers({ radiusBuffer }, singleRadius);
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
			NS_ASSERT(strideInBytes == 0 || strideInBytes >= sizeInBytes);
			NS_ASSERT(sizeInBytes == 1 || sizeInBytes == 2 || sizeInBytes == 4);
			m_native.sbtIndexOffsetBuffer = reinterpret_cast<CUdeviceptr>(offsetBuffer.data());
			m_native.sbtIndexOffsetStrideInBytes = strideInBytes;
			m_native.sbtIndexOffsetSizeInBytes = sizeInBytes;
			return *this;
		}


		/**
		 *	@brief		Set typed SBT index offsets and infer element size and stride.
		 *	@param[in]	offsets - A span of raw integer types representing the SBT index offsets.
		 *	@note		This function deliberately does not change `numSbtRecords`; geometry flags
		 *				and the SBT record count are configured together by `setGeometryFlags()`.
		 */
		template<IndexType Type> BuildInputSpheres & setSbtIndexOffsets(dev::Span<const Type> offsets)
		{
			m_native.sbtIndexOffsetBuffer = reinterpret_cast<CUdeviceptr>(offsets.data());
			m_native.sbtIndexOffsetStrideInBytes = sizeof(Type);
			m_native.sbtIndexOffsetSizeInBytes = sizeof(Type);
			return *this;
		}

	public:

		/**
		 *	@brief		Set per-SBT geometry flags and the corresponding record count.
		 *	@param[in]	flags - Non-empty list containing exactly one flag per SBT record.
		 */
		BuildInputSpheres & setGeometryFlags(ns::Span<const OptixGeometryFlags> flags)
		{
			NS_ASSERT(!flags.empty());
			m_geometryFlags.resize(flags.size());
			std::transform(flags.begin(), flags.end(), m_geometryFlags.begin(), [](OptixGeometryFlags flag) { return static_cast<unsigned int>(flag); });
			m_native.numSbtRecords = static_cast<unsigned int>(flags.size());
			return *this;
		}


		/**
		 *	@brief		Set the SBT record count and replicate one geometry flag for every record.
		 *	@param[in]	numSbtRecords - Number of SBT records available to this build input.
		 *	@param[in]	flag - Geometry behavior assigned to every record.
		 */
		BuildInputSpheres & setGeometryFlags(OptixGeometryFlags flag, unsigned int numSbtRecords = 1)
		{
			NS_ASSERT(numSbtRecords > 0);
			m_geometryFlags.assign(numSbtRecords, static_cast<unsigned int>(flag));
			m_native.numSbtRecords = numSbtRecords;
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
			m_native.primitiveIndexOffset = indexOffset;
			return *this;
		}

public:

		/**
		 *	@brief		Create a native sphere descriptor bound to this wrapper's host arrays.
		 *	@note		The returned descriptor is a lightweight copy. Its host pointers remain
		 *				valid only while this wrapper remains alive and is not modified.
		 */
		[[nodiscard]] OptixBuildInputSphereArray native() const noexcept
		{
			m_native.vertexBuffers = m_vertexBuffers.empty() ? nullptr : m_vertexBuffers.data();
			m_native.radiusBuffers = m_radiusBuffers.empty() ? nullptr : m_radiusBuffers.data();
			m_native.flags = m_geometryFlags.empty() ? nullptr : m_geometryFlags.data();
			return m_native;
		}

	private:

		std::vector<CUdeviceptr>				m_vertexBuffers;
		std::vector<CUdeviceptr>				m_radiusBuffers;
		std::vector<unsigned int>				m_geometryFlags{ OPTIX_GEOMETRY_FLAG_NONE };
		mutable OptixBuildInputSphereArray		m_native{ .numSbtRecords = 1 };
	};

	/*****************************************************************************
	***************************    BuildInputAabbs    ****************************
	*****************************************************************************/

	/**
	 *	@brief		Owning wrapper for `OptixBuildInputCustomPrimitiveArray`.
	 *	@details	The AABB device-address array and geometry-flag array are owned by this
	 *				object. Typed setters infer primitive count and stride; raw setters remain
	 *				available for padded or interleaved AABB layouts.
	 */
	class BuildInputAabbs
	{

	public:

		/**
		 *	@brief		Set raw AABB buffers with explicit primitive count and stride metadata.
		 *	@param[in]	aabbBuffers - One untyped device span for each motion step. Only the
		 *				device address is retained; the host-side address array is owned by this wrapper.
		 *	@param[in]	numPrimitives - The number of AABBs in each buffer.
		 *	@param[in]	strideInBytes - The stride between consecutive AABBs in bytes. Default is 0 (tightly packed).
		 */
		BuildInputAabbs & setAabbBuffers(ns::Span<const dev::Span<const ns::byte>> aabbBuffers, size_t numPrimitives, size_t strideInBytes = 0)
		{
			m_aabbBuffers.resize(aabbBuffers.size());
			for (size_t i = 0; i < aabbBuffers.size(); ++i)
				m_aabbBuffers[i] = reinterpret_cast<CUdeviceptr>(aabbBuffers[i].data());
			m_native.numPrimitives = static_cast<unsigned int>(numPrimitives);
			m_native.strideInBytes = static_cast<unsigned int>(strideInBytes);
			return *this;
		}


		/**
		 *	@brief		Set typed AABB buffers and infer primitive count and stride.
		 *	@param[in]	aabbBuffers - One typed AABB device span per motion step.
		 *	@note		Every motion step must contain the same number of primitives.
		 */
		template<typename Type> requires (std::same_as<std::remove_cv_t<Type>, Aabb> || std::same_as<std::remove_cv_t<Type>, OptixAabb>)
		BuildInputAabbs & setAabbBuffers(ns::Span<const dev::Span<const Type>> aabbBuffers)
		{
			for (const auto & aabbBuffer : aabbBuffers)
				NS_ASSERT(aabbBuffers.empty() || aabbBuffer.size() == aabbBuffers.front().size());

			m_aabbBuffers.resize(aabbBuffers.size());
			for (size_t i = 0; i < aabbBuffers.size(); ++i)
				m_aabbBuffers[i] = reinterpret_cast<CUdeviceptr>(aabbBuffers[i].data());
			m_native.numPrimitives = aabbBuffers.empty() ? 0u : static_cast<unsigned int>(aabbBuffers.front().size());
			m_native.strideInBytes = static_cast<unsigned int>(sizeof(Type));
			return *this;
		}


		/**
		 *	@brief		Set one typed AABB buffer and infer its primitive count and stride.
		 *	@param[in]	aabbBuffer - Typed AABB storage for a single motion step.
		 */
		template<typename Type> requires (std::same_as<std::remove_cv_t<Type>, Aabb> || std::same_as<std::remove_cv_t<Type>, OptixAabb>)
		BuildInputAabbs & setAabbBuffer(dev::Span<const Type> aabbBuffer)
		{
			return setAabbBuffers<Type>({ aabbBuffer });
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
			NS_ASSERT(sizeInBytes == 1 || sizeInBytes == 2 || sizeInBytes == 4);
			NS_ASSERT(strideInBytes == 0 || strideInBytes >= sizeInBytes);
			m_native.sbtIndexOffsetBuffer = reinterpret_cast<CUdeviceptr>(offsetBuffer.data());
			m_native.sbtIndexOffsetStrideInBytes = strideInBytes;
			m_native.sbtIndexOffsetSizeInBytes = sizeInBytes;
			return *this;
		}


		/**
		 *	@brief		Set typed SBT index offsets and infer element size and stride.
		 *	@param[in]	offsets - A span of raw integer types representing the SBT index offsets.
		 *	@note		This function deliberately does not change `numSbtRecords`; geometry flags
		 *				and the SBT record count are configured together by `setGeometryFlags()`.
		 */
		template<IndexType Type> BuildInputAabbs & setSbtIndexOffsets(dev::Span<const Type> offsets)
		{
			m_native.sbtIndexOffsetBuffer = reinterpret_cast<CUdeviceptr>(offsets.data());
			m_native.sbtIndexOffsetStrideInBytes = sizeof(Type);
			m_native.sbtIndexOffsetSizeInBytes = sizeof(Type);
			return *this;
		}

	public:

		/**
		 *	@brief		Set per-SBT geometry flags and the corresponding record count.
		 *	@param[in]	flags - Non-empty list containing exactly one flag per SBT record.
		 */
		BuildInputAabbs & setGeometryFlags(ns::Span<const OptixGeometryFlags> flags)
		{
			NS_ASSERT(!flags.empty());
			m_geometryFlags.resize(flags.size());
			std::transform(flags.begin(), flags.end(), m_geometryFlags.begin(), [](OptixGeometryFlags flag) { return static_cast<unsigned int>(flag); });
			m_native.numSbtRecords = static_cast<unsigned int>(flags.size());
			return *this;
		}


		/**
		 *	@brief		Set the SBT record count and replicate one geometry flag for every record.
		 *	@param[in]	numSbtRecords - Number of SBT records available to this build input.
		 *	@param[in]	flag - Geometry behavior assigned to every record.
		 */
		BuildInputAabbs & setGeometryFlags(OptixGeometryFlags flag, unsigned int numSbtRecords = 1)
		{
			NS_ASSERT(numSbtRecords > 0);
			m_geometryFlags.assign(numSbtRecords, static_cast<unsigned int>(flag));
			m_native.numSbtRecords = numSbtRecords;
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
			m_native.primitiveIndexOffset = indexOffset;
			return *this;
		}

	public:

		/**
		 *	@brief		Create a native custom-primitive descriptor bound to this wrapper's host arrays.
		 *	@note		The returned descriptor is a lightweight copy. Its host pointers remain
		 *				valid only while this wrapper remains alive and is not modified.
		 */
		[[nodiscard]] OptixBuildInputCustomPrimitiveArray native() const noexcept
		{
			m_native.aabbBuffers = m_aabbBuffers.empty() ? nullptr : m_aabbBuffers.data();
			m_native.flags = m_geometryFlags.empty() ? nullptr : m_geometryFlags.data();
			return m_native;
		}

	private:

		std::vector<CUdeviceptr>						m_aabbBuffers;
		std::vector<unsigned int>						m_geometryFlags{ OPTIX_GEOMETRY_FLAG_NONE };
		mutable OptixBuildInputCustomPrimitiveArray		m_native{ .numSbtRecords = 1 };
	};
}
