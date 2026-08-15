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

#include <photon/build_inputs.h>

/*********************************************************************************
****************************    test_build_inputs    *****************************
*********************************************************************************/

void test_build_inputs()
{
	//	Triangle build input.
	pt::BuildInputTriangles triangles;
	assert(triangles.vertexBuffers == nullptr);
	assert(triangles.numVertices == 0);
	assert(triangles.vertexFormat == OPTIX_VERTEX_FORMAT_NONE);
	assert(triangles.vertexStrideInBytes == 0);
	assert(triangles.indexBuffer == 0);
	assert(triangles.indexFormat == OPTIX_INDICES_FORMAT_NONE);
	assert(triangles.numIndexTriplets == 0);
	assert(triangles.indexStrideInBytes == 0);
	assert(triangles.numSbtRecords == 1);
	assert(triangles.sbtIndexOffsetBuffer == 0);
	assert(triangles.sbtIndexOffsetSizeInBytes == 0);
	assert(triangles.sbtIndexOffsetStrideInBytes == 0);
	assert(triangles.primitiveIndexOffset == 0);

	CUdeviceptr triangleVertexBuffers[] = { 1, 2 };
	assert(&triangles.setVertexBuffers(triangleVertexBuffers, pt::VertexFormat::Float3, 16, 16) == &triangles);
	assert(triangles.vertexBuffers == triangleVertexBuffers);
	assert(triangles.numVertices == 16);
	assert(triangles.vertexFormat == OPTIX_VERTEX_FORMAT_FLOAT3);
	assert(triangles.vertexStrideInBytes == 16);

	//	Scalar indices are grouped into triplets.
	unsigned short triangleIndices[] = { 0, 1, 2, 2, 3, 0 };
	assert(&triangles.setIndexBuffer(ns::Span<const unsigned short>(triangleIndices)) == &triangles);
	assert(triangles.indexBuffer == reinterpret_cast<CUdeviceptr>(triangleIndices));
	assert(triangles.indexFormat == OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3);
	assert(triangles.numIndexTriplets == 2);
	assert(triangles.indexStrideInBytes == 0);

	//	Triplet types preserve their element stride.
	ns::uint3 triangleIndexTriplets[] = { { 0, 1, 2 }, { 2, 3, 0 } };
	assert(&triangles.setIndexBuffer(ns::Span<const ns::uint3>(triangleIndexTriplets)) == &triangles);
	assert(triangles.indexBuffer == reinterpret_cast<CUdeviceptr>(triangleIndexTriplets));
	assert(triangles.indexFormat == OPTIX_INDICES_FORMAT_UNSIGNED_INT3);
	assert(triangles.numIndexTriplets == 2);
	assert(triangles.indexStrideInBytes == sizeof(ns::uint3));

	ns::byte stridedTriangleIndices[24] = {};
	assert(&triangles.setIndexBuffer(stridedTriangleIndices, pt::IndicesFormat::Ushort3, 8) == &triangles);
	assert(triangles.indexBuffer == reinterpret_cast<CUdeviceptr>(stridedTriangleIndices));
	assert(triangles.indexFormat == OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3);
	assert(triangles.numIndexTriplets == 3);
	assert(triangles.indexStrideInBytes == 8);

	unsigned short triangleSbtIndices[] = { 0, 1, 2, 3 };
	assert(&triangles.setSbtIndexOffsets(ns::Span<const unsigned short>(triangleSbtIndices)) == &triangles);
	assert(triangles.numSbtRecords == 4);
	assert(triangles.sbtIndexOffsetBuffer == reinterpret_cast<CUdeviceptr>(triangleSbtIndices));
	assert(triangles.sbtIndexOffsetSizeInBytes == sizeof(unsigned short));
	assert(triangles.sbtIndexOffsetStrideInBytes == 0);

	ns::byte stridedTriangleSbtIndices[12] = {};
	assert(&triangles.setSbtIndexOffsets(stridedTriangleSbtIndices, 2, 3) == &triangles);
	assert(triangles.numSbtRecords == 4);
	assert(triangles.sbtIndexOffsetBuffer == reinterpret_cast<CUdeviceptr>(stridedTriangleSbtIndices));
	assert(triangles.sbtIndexOffsetSizeInBytes == 2);
	assert(triangles.sbtIndexOffsetStrideInBytes == 3);

	assert(&triangles.setPrimitiveIndexOffset(11) == &triangles);
	assert(triangles.primitiveIndexOffset == 11);

	//	Sphere build input.
	pt::BuildInputSpheres spheres;
	assert(spheres.vertexBuffers == nullptr);
	assert(spheres.vertexStrideInBytes == 0);
	assert(spheres.numVertices == 0);
	assert(spheres.radiusBuffers == nullptr);
	assert(spheres.radiusStrideInBytes == 0);
	assert(spheres.singleRadius == 0);
	assert(spheres.numSbtRecords == 1);
	assert(spheres.sbtIndexOffsetBuffer == 0);
	assert(spheres.sbtIndexOffsetSizeInBytes == 0);
	assert(spheres.sbtIndexOffsetStrideInBytes == 0);
	assert(spheres.primitiveIndexOffset == 0);

	CUdeviceptr vertexBuffers[] = { 1, 2 };
	assert(&spheres.setVertexBuffers(vertexBuffers, 32, 16) == &spheres);
	assert(spheres.vertexBuffers == vertexBuffers);
	assert(spheres.vertexStrideInBytes == 16);
	assert(spheres.numVertices == 32);

	CUdeviceptr radiusBuffers[] = { 3, 4 };
	assert(&spheres.setRadiusBuffers(radiusBuffers, true, 8) == &spheres);
	assert(spheres.radiusBuffers == radiusBuffers);
	assert(spheres.radiusStrideInBytes == 8);
	assert(spheres.singleRadius == 1);

	unsigned short sbtIndices[] = { 0, 1, 2, 3 };
	assert(&spheres.setSbtIndexOffsets(ns::Span<const unsigned short>(sbtIndices)) == &spheres);
	assert(spheres.numSbtRecords == 4);
	assert(spheres.sbtIndexOffsetBuffer == reinterpret_cast<CUdeviceptr>(sbtIndices));
	assert(spheres.sbtIndexOffsetSizeInBytes == sizeof(unsigned short));
	assert(spheres.sbtIndexOffsetStrideInBytes == 0);

	ns::byte stridedSbtIndices[12] = {};
	assert(&spheres.setSbtIndexOffsets(stridedSbtIndices, 2, 3) == &spheres);
	assert(spheres.numSbtRecords == 4);
	assert(spheres.sbtIndexOffsetBuffer == reinterpret_cast<CUdeviceptr>(stridedSbtIndices));
	assert(spheres.sbtIndexOffsetSizeInBytes == 2);
	assert(spheres.sbtIndexOffsetStrideInBytes == 3);

	assert(&spheres.setPrimitiveIndexOffset(17) == &spheres);
	assert(spheres.primitiveIndexOffset == 17);

#if OPTIX_VERSION >= 70100
	//	Curve build input.
	pt::BuildInputCurves curves;
	assert(curves.curveType == static_cast<OptixPrimitiveType>(0));
	assert(curves.numPrimitives == 0);
	assert(curves.vertexBuffers == nullptr);
	assert(curves.numVertices == 0);
	assert(curves.vertexStrideInBytes == 0);
	assert(curves.widthBuffers == nullptr);
	assert(curves.widthStrideInBytes == 0);
	assert(curves.indexBuffer == 0);
	assert(curves.indexStrideInBytes == 0);
	assert(curves.flag == 0);
	assert(curves.primitiveIndexOffset == 0);
	assert(curves.endcapFlags == 0);

	assert(&curves.setCurveType(OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE) == &curves);
	assert(curves.curveType == OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE);

	CUdeviceptr curveVertexBuffers[] = { 5, 6 };
	assert(&curves.setVertexBuffers(curveVertexBuffers, 48, 16) == &curves);
	assert(curves.vertexBuffers == curveVertexBuffers);
	assert(curves.numVertices == 48);
	assert(curves.vertexStrideInBytes == 16);

	CUdeviceptr curveWidthBuffers[] = { 7, 8 };
	assert(&curves.setWidthBuffers(curveWidthBuffers, 8) == &curves);
	assert(curves.widthBuffers == curveWidthBuffers);
	assert(curves.widthStrideInBytes == 8);

	unsigned int curveIndices[] = { 0, 1, 2, 3 };
	assert(&curves.setIndexBuffer(ns::Span<const unsigned int>(curveIndices)) == &curves);
	assert(curves.numPrimitives == 4);
	assert(curves.indexBuffer == reinterpret_cast<CUdeviceptr>(curveIndices));
	assert(curves.indexStrideInBytes == 0);

	ns::byte stridedCurveIndices[32] = {};
	assert(&curves.setIndexBuffer(stridedCurveIndices, 8) == &curves);
	assert(curves.numPrimitives == 4);
	assert(curves.indexBuffer == reinterpret_cast<CUdeviceptr>(stridedCurveIndices));
	assert(curves.indexStrideInBytes == 8);

	assert(&curves.setGeometryFlags(OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT) == &curves);
	assert(curves.flag == OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);
	assert(&curves.setEndcapFlags(OPTIX_CURVE_ENDCAP_ON) == &curves);
	assert(curves.endcapFlags == OPTIX_CURVE_ENDCAP_ON);
	assert(&curves.setPrimitiveIndexOffset(23) == &curves);
	assert(curves.primitiveIndexOffset == 23);
#endif

	//	AABB build input.
	pt::BuildInputAabbs aabbs;
	assert(aabbs.aabbBuffers == nullptr);
	assert(aabbs.numPrimitives == 0);
	assert(aabbs.strideInBytes == 0);
	assert(aabbs.numSbtRecords == 1);
	assert(aabbs.sbtIndexOffsetBuffer == 0);
	assert(aabbs.sbtIndexOffsetSizeInBytes == 0);
	assert(aabbs.sbtIndexOffsetStrideInBytes == 0);
	assert(aabbs.primitiveIndexOffset == 0);

	CUdeviceptr aabbBuffers[] = { 9, 10 };
	assert(&aabbs.setAabbBuffers(aabbBuffers, 24, sizeof(pt::Aabb)) == &aabbs);
	assert(aabbs.aabbBuffers == aabbBuffers);
	assert(aabbs.numPrimitives == 24);
	assert(aabbs.strideInBytes == sizeof(pt::Aabb));

	unsigned char aabbSbtIndices[] = { 0, 1, 2 };
	assert(&aabbs.setSbtIndexOffsets(ns::Span<const unsigned char>(aabbSbtIndices)) == &aabbs);
	assert(aabbs.numSbtRecords == 3);
	assert(aabbs.sbtIndexOffsetBuffer == reinterpret_cast<CUdeviceptr>(aabbSbtIndices));
	assert(aabbs.sbtIndexOffsetSizeInBytes == sizeof(unsigned char));
	assert(aabbs.sbtIndexOffsetStrideInBytes == 0);

	ns::byte stridedAabbSbtIndices[16] = {};
	assert(&aabbs.setSbtIndexOffsets(stridedAabbSbtIndices, 2, 4) == &aabbs);
	assert(aabbs.numSbtRecords == 4);
	assert(aabbs.sbtIndexOffsetBuffer == reinterpret_cast<CUdeviceptr>(stridedAabbSbtIndices));
	assert(aabbs.sbtIndexOffsetSizeInBytes == 2);
	assert(aabbs.sbtIndexOffsetStrideInBytes == 4);

	assert(&aabbs.setPrimitiveIndexOffset(29) == &aabbs);
	assert(aabbs.primitiveIndexOffset == 29);
}
