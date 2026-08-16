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

#include <nucleus/array_1d.h>
#include <photon/build_inputs.h>

/*********************************************************************************
****************************    test_build_inputs    *****************************
*********************************************************************************/

void test_build_inputs()
{
	//	Triangle build input.
	pt::BuildInputTriangles triangles;
	assert(triangles.numVertices == 0);
	assert(triangles.indexBuffer == 0);
	assert(triangles.numSbtRecords == 1);
	assert(triangles.numIndexTriplets == 0);
	assert(triangles.indexStrideInBytes == 0);
	assert(triangles.vertexBuffers == nullptr);
	assert(triangles.vertexStrideInBytes == 0);
	assert(triangles.sbtIndexOffsetBuffer == 0);
	assert(triangles.primitiveIndexOffset == 0);
	assert(triangles.sbtIndexOffsetSizeInBytes == 0);
	assert(triangles.sbtIndexOffsetStrideInBytes == 0);
	assert(triangles.vertexFormat == OPTIX_VERTEX_FORMAT_NONE);
	assert(triangles.indexFormat == OPTIX_INDICES_FORMAT_NONE);

	ns::Array<CUdeviceptr> triangleVertexBuffers;
	assert(&triangles.setVertexBuffers({ triangleVertexBuffers.data(), triangleVertexBuffers.size() }, pt::VertexFormat::Float3, 16) == &triangles);
	assert(triangles.vertexFormat == OPTIX_VERTEX_FORMAT_FLOAT3);
	assert(triangles.vertexStrideInBytes == 0);
	assert(triangles.numVertices == 16);

	//	Scalar indices are grouped into triplets.
	ns::Array<unsigned short> triangleIndices;
	assert(&triangles.setIndexBuffer(triangleIndices.span()) == &triangles);
	assert(triangles.indexFormat == OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3);
	assert(triangles.numIndexTriplets == triangleIndices.size());
	assert(triangles.indexStrideInBytes == 0);

	//	Triplet types preserve their element stride.
	ns::Array<ns::uint3> triangleIndexTriplets;
	assert(&triangles.setIndexBuffer(triangleIndexTriplets.span()) == &triangles);
	assert(triangles.indexFormat == OPTIX_INDICES_FORMAT_UNSIGNED_INT3);
	assert(triangles.numIndexTriplets == triangleIndexTriplets.size());
	assert(triangles.indexStrideInBytes == sizeof(ns::uint3));

	ns::Array<ns::byte> stridedTriangleIndices;
	assert(&triangles.setIndexBuffer(stridedTriangleIndices.span(), pt::IndicesFormat::Ushort3, 8) == &triangles);
	assert(triangles.indexFormat == OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3);
	assert(triangles.numIndexTriplets == stridedTriangleIndices.size());
	assert(triangles.indexStrideInBytes == 8);

	ns::Array<unsigned short> triangleSbtIndices;
	assert(&triangles.setSbtIndexOffsets(triangleSbtIndices) == &triangles);
	assert(triangles.sbtIndexOffsetSizeInBytes == sizeof(unsigned short));
	assert(triangles.numSbtRecords == triangleSbtIndices.size());
	assert(triangles.sbtIndexOffsetStrideInBytes == 0);

	ns::Array<ns::byte> stridedTriangleSbtIndices;
	assert(&triangles.setSbtIndexOffsets(stridedTriangleSbtIndices, 2, 3) == &triangles);
	assert(triangles.numSbtRecords == stridedTriangleSbtIndices.size());
	assert(triangles.sbtIndexOffsetStrideInBytes == 3);
	assert(triangles.sbtIndexOffsetSizeInBytes == 2);

	assert(&triangles.setPrimitiveIndexOffset(11) == &triangles);
	assert(triangles.primitiveIndexOffset == 11);

	//	Sphere build input.
	pt::BuildInputSpheres spheres;
	assert(spheres.numVertices == 0);
	assert(spheres.singleRadius == 0);
	assert(spheres.numSbtRecords == 1);
	assert(spheres.vertexBuffers == nullptr);
	assert(spheres.radiusBuffers == nullptr);
	assert(spheres.radiusStrideInBytes == 0);
	assert(spheres.vertexStrideInBytes == 0);
	assert(spheres.sbtIndexOffsetBuffer == 0);
	assert(spheres.primitiveIndexOffset == 0);
	assert(spheres.sbtIndexOffsetSizeInBytes == 0);
	assert(spheres.sbtIndexOffsetStrideInBytes == 0);

	ns::Array<CUdeviceptr> vertexBuffers;
	assert(&spheres.setVertexBuffers({ vertexBuffers.data(), vertexBuffers.size() }, 32, 16) == &spheres);
	assert(spheres.vertexStrideInBytes == 16);
	assert(spheres.numVertices == 32);

	ns::Array<CUdeviceptr> radiusBuffers;
	assert(&spheres.setRadiusBuffers({ radiusBuffers.data(), radiusBuffers.size() }, true, 8) == &spheres);
	assert(spheres.radiusStrideInBytes == 8);
	assert(spheres.singleRadius == 1);

	ns::Array<unsigned short> sbtIndices;
	assert(&spheres.setSbtIndexOffsets(sbtIndices) == &spheres);
	assert(spheres.sbtIndexOffsetSizeInBytes == sizeof(unsigned short));
	assert(spheres.numSbtRecords == sbtIndices.size());
	assert(spheres.sbtIndexOffsetStrideInBytes == 0);

	ns::Array<ns::byte> stridedSbtIndices;
	assert(&spheres.setSbtIndexOffsets(stridedSbtIndices, 2, 3) == &spheres);
	assert(spheres.numSbtRecords == stridedSbtIndices.size());
	assert(spheres.sbtIndexOffsetStrideInBytes == 3);
	assert(spheres.sbtIndexOffsetSizeInBytes == 2);

	assert(&spheres.setPrimitiveIndexOffset(17) == &spheres);
	assert(spheres.primitiveIndexOffset == 17);

#if OPTIX_VERSION >= 70100
	//	Curve build input.
	pt::BuildInputCurves curves;
	assert(curves.flag == 0);
	assert(curves.numVertices == 0);
	assert(curves.indexBuffer == 0);
	assert(curves.endcapFlags == 0);
	assert(curves.numPrimitives == 0);
	assert(curves.widthStrideInBytes == 0);
	assert(curves.indexStrideInBytes == 0);
	assert(curves.widthBuffers == nullptr);
	assert(curves.vertexBuffers == nullptr);
	assert(curves.vertexStrideInBytes == 0);
	assert(curves.primitiveIndexOffset == 0);
	assert(curves.curveType == static_cast<OptixPrimitiveType>(0));

	assert(&curves.setCurveType(OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE) == &curves);
	assert(curves.curveType == OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE);

	ns::Array<CUdeviceptr> curveVertexBuffers;
	assert(&curves.setVertexBuffers({ curveVertexBuffers.data(), curveVertexBuffers.size() }, 48, 16) == &curves);
	assert(curves.vertexStrideInBytes == 16);
	assert(curves.numVertices == 48);

	ns::Array<CUdeviceptr> curveWidthBuffers;
	assert(&curves.setWidthBuffers({ curveWidthBuffers.data(), curveWidthBuffers.size()}, 8) == &curves);
	assert(curves.widthStrideInBytes == 8);

	ns::Array<unsigned int> curveIndices;
	assert(&curves.setIndexBuffer(curveIndices) == &curves);
	assert(curves.numPrimitives == curveIndices.size());
	assert(curves.indexStrideInBytes == 0);

	ns::Array<ns::byte> stridedCurveIndices;
	assert(&curves.setIndexBuffer(stridedCurveIndices, 8) == &curves);
	assert(curves.numPrimitives == stridedCurveIndices.size());
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
	assert(aabbs.numPrimitives == 0);
	assert(aabbs.strideInBytes == 0);
	assert(aabbs.numSbtRecords == 1);
	assert(aabbs.aabbBuffers == nullptr);
	assert(aabbs.sbtIndexOffsetBuffer == 0);
	assert(aabbs.primitiveIndexOffset == 0);
	assert(aabbs.sbtIndexOffsetSizeInBytes == 0);
	assert(aabbs.sbtIndexOffsetStrideInBytes == 0);

	ns::Array<CUdeviceptr> aabbBuffers;
	assert(&aabbs.setAabbBuffers({ aabbBuffers.data(), aabbBuffers.size() }, 24, sizeof(pt::Aabb)) == &aabbs);
	assert(aabbs.strideInBytes == sizeof(pt::Aabb));
	assert(aabbs.numPrimitives == 24);

	ns::Array<unsigned char> aabbSbtIndices;
	assert(&aabbs.setSbtIndexOffsets(aabbSbtIndices) == &aabbs);
	assert(aabbs.sbtIndexOffsetSizeInBytes == sizeof(unsigned char));
	assert(aabbs.numSbtRecords == aabbSbtIndices.size());
	assert(aabbs.sbtIndexOffsetStrideInBytes == 0);

	ns::Array<ns::byte> stridedAabbSbtIndices;
	assert(&aabbs.setSbtIndexOffsets(stridedAabbSbtIndices, 2, 4) == &aabbs);
	assert(aabbs.numSbtRecords == stridedAabbSbtIndices.size());
	assert(aabbs.sbtIndexOffsetStrideInBytes == 4);
	assert(aabbs.sbtIndexOffsetSizeInBytes == 2);

	assert(&aabbs.setPrimitiveIndexOffset(29) == &aabbs);
	assert(aabbs.primitiveIndexOffset == 29);
}
