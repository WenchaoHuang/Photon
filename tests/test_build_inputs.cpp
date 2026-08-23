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

#include <cassert>
#include <cstdint>
#include <photon/build_inputs.h>

/*********************************************************************************
****************************    test_build_inputs    *****************************
*********************************************************************************/

void test_build_inputs()
{
	const auto radiusAddress = reinterpret_cast<const float*>(std::uintptr_t{ 0x2000 });
	const auto aabbAddress = reinterpret_cast<const pt::Aabb*>(std::uintptr_t{ 0x5000 });
	const auto vertexAddress = reinterpret_cast<const ns::float3*>(std::uintptr_t{ 0x1000 });
	const auto indexAddress = reinterpret_cast<const unsigned int*>(std::uintptr_t{ 0x3000 });
	const auto offsetAddress = reinterpret_cast<const unsigned short*>(std::uintptr_t{ 0x4000 });

	const dev::Span<const ns::float3> vertices(vertexAddress, 12);
	const dev::Span<const float> radii(radiusAddress, 12);
	const dev::Span<const unsigned int> indices(indexAddress, 18);
	const dev::Span<const unsigned short> offsets(offsetAddress, 6);
	const dev::Span<const pt::Aabb> aabbs(aabbAddress, 7);
	const dev::Span<const ns::byte> rawRadii[] = { ns::as_bytes(radii) };
	const dev::Span<const ns::byte> rawAabbs[] = { ns::as_bytes(aabbs) };
	const dev::Span<const ns::byte> rawVertices[] = { ns::as_bytes(vertices) };

	// Typed triangle inputs infer all format, count and stride metadata.
	pt::BuildInputTriangles triangles;
	triangles.setIndexBuffer(indices);
	triangles.setVertexBuffer(vertices);
	triangles.setSbtIndexOffsets(offsets);
	triangles.setPrimitiveIndexOffset(11);
	triangles.setGeometryFlags(OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT, 3);
	auto triangleNative = triangles.native();
	assert(triangleNative.vertexBuffers != nullptr);
	assert(triangleNative.vertexBuffers[0] == reinterpret_cast<CUdeviceptr>(vertexAddress));
	assert(triangleNative.vertexFormat == OPTIX_VERTEX_FORMAT_FLOAT3);
	assert(triangleNative.vertexStrideInBytes == sizeof(ns::float3));
	assert(triangleNative.numVertices == vertices.size());
	assert(triangleNative.indexFormat == OPTIX_INDICES_FORMAT_UNSIGNED_INT3);
	assert(triangleNative.indexStrideInBytes == 3 * sizeof(unsigned int));
	assert(triangleNative.numIndexTriplets == indices.size() / 3);
	assert(triangleNative.numSbtRecords == 3);
	assert(triangleNative.flags[0] == OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);
	assert(triangleNative.sbtIndexOffsetSizeInBytes == sizeof(unsigned short));
	assert(triangleNative.sbtIndexOffsetStrideInBytes == sizeof(unsigned short));
	assert(triangleNative.primitiveIndexOffset == 11);

	// A copied wrapper rebinds native host pointers to its own vector storage.
	pt::BuildInputTriangles copiedTriangles = triangles;
	auto copiedTriangleNative = copiedTriangles.native();
	assert(copiedTriangleNative.vertexBuffers != triangleNative.vertexBuffers);
	assert(copiedTriangleNative.flags != triangleNative.flags);
	assert(copiedTriangleNative.vertexBuffers[0] == triangleNative.vertexBuffers[0]);
	assert(copiedTriangleNative.flags[0] == triangleNative.flags[0]);

	// Raw interfaces preserve explicitly supplied layout metadata.
	const dev::Span<const ns::byte> rawIndices(reinterpret_cast<const ns::byte*>(indexAddress), 32);
	triangles.setIndexBuffer(rawIndices, pt::IndicesFormat::Ushort3, 8);
	triangleNative = triangles.native();
	assert(triangleNative.indexFormat == OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3);
	assert(triangleNative.indexStrideInBytes == 8);
	assert(triangleNative.numIndexTriplets == 4);
	triangles.setVertexBuffers(rawVertices, pt::VertexFormat::Float3, 9, 16);
	triangleNative = triangles.native();
	assert(triangleNative.vertexBuffers[0] == reinterpret_cast<CUdeviceptr>(vertexAddress));
	assert(triangleNative.numVertices == 9);
	assert(triangleNative.vertexStrideInBytes == 16);

	// Setting an offset buffer is independent from the SBT record count.
	triangles.setSbtIndexOffsets(offsets);
	triangles.setGeometryFlags(OPTIX_GEOMETRY_FLAG_NONE, 5);
	assert(triangles.native().numSbtRecords == 5);

	// Sphere inputs own both host-side device-pointer arrays and the flag array.
	pt::BuildInputSpheres spheres;
	spheres.setRadiusBuffer(radii);
	spheres.setVertexBuffer(vertices);
	spheres.setSbtIndexOffsets(offsets);
	spheres.setPrimitiveIndexOffset(17);
	spheres.setGeometryFlags(OPTIX_GEOMETRY_FLAG_NONE, 2);
	auto sphereNative = spheres.native();
	assert(sphereNative.vertexBuffers[0] == reinterpret_cast<CUdeviceptr>(vertexAddress));
	assert(sphereNative.radiusBuffers[0] == reinterpret_cast<CUdeviceptr>(radiusAddress));
	assert(sphereNative.vertexStrideInBytes == sizeof(ns::float3));
	assert(sphereNative.radiusStrideInBytes == sizeof(float));
	assert(sphereNative.numVertices == vertices.size());
	assert(sphereNative.numSbtRecords == 2);
	assert(sphereNative.sbtIndexOffsetSizeInBytes == sizeof(unsigned short));
	assert(sphereNative.sbtIndexOffsetStrideInBytes == sizeof(unsigned short));
	assert(sphereNative.primitiveIndexOffset == 17);
	spheres.setVertexBuffers(rawVertices, 9, 16);
	spheres.setRadiusBuffers(rawRadii, false, 8);
	sphereNative = spheres.native();
	assert(sphereNative.numVertices == 9);
	assert(sphereNative.vertexStrideInBytes == 16);
	assert(sphereNative.radiusStrideInBytes == 8);

#if OPTIX_VERSION >= 70100
	// Curve inputs infer typed buffer strides while retaining raw index control.
	const auto curveVertexAddress = reinterpret_cast<const ns::float4*>(std::uintptr_t{ 0x6000 });
	const dev::Span<const ns::float4> curveVertices(curveVertexAddress, 9);
	const dev::Span<const ns::byte> rawCurveVertices[] = { ns::as_bytes(curveVertices) };
	pt::BuildInputCurves curves;
	curves.setWidthBuffer(radii);
	curves.setIndexBuffer(indices);
	curves.setPrimitiveIndexOffset(23);
	curves.setVertexBuffers(curveVertices);
	curves.setEndcapFlags(OPTIX_CURVE_ENDCAP_ON);
	curves.setGeometryFlags(OPTIX_GEOMETRY_FLAG_NONE);
	curves.setCurveType(OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE);
	auto curveNative = curves.native();
	assert(curveNative.vertexBuffers[0] == reinterpret_cast<CUdeviceptr>(curveVertexAddress));
	assert(curveNative.widthBuffers[0] == reinterpret_cast<CUdeviceptr>(radiusAddress));
	assert(curveNative.vertexStrideInBytes == sizeof(ns::float4));
	assert(curveNative.widthStrideInBytes == sizeof(float));
	assert(curveNative.indexStrideInBytes == sizeof(unsigned int));
	assert(curveNative.numVertices == curveVertices.size());
	assert(curveNative.numPrimitives == indices.size());
	assert(curveNative.flag == OPTIX_GEOMETRY_FLAG_NONE);
	assert(curveNative.endcapFlags == OPTIX_CURVE_ENDCAP_ON);
	assert(curveNative.primitiveIndexOffset == 23);
	curves.setVertexBuffers(rawCurveVertices, 7, 20);
	curves.setWidthBuffers(rawRadii, 8);
	curveNative = curves.native();
	assert(curveNative.numVertices == 7);
	assert(curveNative.vertexStrideInBytes == 20);
	assert(curveNative.widthStrideInBytes == 8);
#endif

	// Typed AABB inputs infer primitive count and element stride.
	pt::BuildInputAabbs customPrimitives;
	customPrimitives.setAabbBuffer(aabbs);
	customPrimitives.setSbtIndexOffsets(offsets);
	customPrimitives.setPrimitiveIndexOffset(29);
	customPrimitives.setGeometryFlags(OPTIX_GEOMETRY_FLAG_NONE, 4);
	auto aabbNative = customPrimitives.native();
	assert(aabbNative.aabbBuffers[0] == reinterpret_cast<CUdeviceptr>(aabbAddress));
	assert(aabbNative.numPrimitives == aabbs.size());
	assert(aabbNative.strideInBytes == sizeof(pt::Aabb));
	assert(aabbNative.numSbtRecords == 4);
	assert(aabbNative.sbtIndexOffsetSizeInBytes == sizeof(unsigned short));
	assert(aabbNative.sbtIndexOffsetStrideInBytes == sizeof(unsigned short));
	assert(aabbNative.primitiveIndexOffset == 29);
	customPrimitives.setAabbBuffers(rawAabbs, 5, 32);
	aabbNative = customPrimitives.native();
	assert(aabbNative.numPrimitives == 5);
	assert(aabbNative.strideInBytes == 32);
}
