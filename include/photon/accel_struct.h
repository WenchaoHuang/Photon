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
#include <nucleus/array_proxy.h>
#include <nucleus/buffer_view.h>
#include <nucleus/vector_types.h>

namespace PHOTON_NAMESPACE
{
	struct NS_ALIGN(8) Aabb { ns::float3 lower, upper; };
	struct NS_ALIGN(16) Mat4x4 { ns::float4 rows[4]; };

	/*****************************************************************************
	*****************************    AccelStruct    ******************************
	*****************************************************************************/

	//	An abstract class representing an acceleration structure.
	class AccelStruct
	{

	public:

		//	Default constructor.
		AccelStruct() {}

		//	Virtual destructor.
		virtual ~AccelStruct() {}

	public:

		//	Enum defining different subtypes of acceleration structures.
		enum SubType
		{
			Geometry,		//	Geometry acceleration structure (GAS).
			Instance,		//	Instance acceleration structure (IAS).
		};

		//	Pure virtual function to check if the acceleration structure is empty.
		virtual bool empty() const = 0;

		//	Pure virtual function to check if updates are allowed on the acceleration structure.
		virtual bool allowUpdate() const = 0;

		//	Pure virtual function to retrieve the subtype of the acceleration structure.
		virtual SubType subType() const = 0;

		//	Pure virtual function to retrieve the total number of SbtRecords.
		virtual unsigned int numSbtRecords() const = 0;

		//	Pure virtual function to retrieve the handle of Optix acceleration structure.
		virtual OptixTraversableHandle handle() const = 0;

		//	Pure virtual function to retrieve the context associated with the acceleration structure.
		virtual std::shared_ptr<class DeviceContext> deviceContext() const = 0;

		//	Pure virtual function to rebuild the acceleration structure.
		virtual void rebuild(ns::Stream & stream) = 0;

		/**
		 *	@brief		Pure virtual function to refit the acceleration structure.
		 *	@warning	Only the device pointers and/or their buffer content may be changed.
		 */
		virtual void refit(ns::Stream & stream) = 0;
	};

	/*****************************************************************************
	***************************    GeomAccelStruct    ****************************
	*****************************************************************************/

	//	An abstract class representing an geometry acceleration structure.
	class GeomAccelStruct : virtual public AccelStruct
	{

	public:

		//	Enum defining different types of primitives supported by the acceleration structure.
		enum PrimitiveType
		{
			Triangle,		//	Geometry acceleration structure containing built-in triangles.
			Sphere,		//	Geometry acceleration structure containing built-in spheres.
			Curve,			//	Geometry acceleration structure containing built-in curve primitives.
			AABB,			//	Geometry acceleration structure containing custom primitives.
		};

		//	Geometry flags describing the primitive behavior.
		enum GeomFlags
		{
			None							= 0,			//	No flags set.
			DisableAnyhit					= 1u << 0,		//	Disables the invocation of the anyhit program. Can be overridden by OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT and OPTIX_RAY_FLAG_ENFORCE_ANYHIT.
			RequireSingleAnyhitCall			= 1u << 1,		//	If set, an intersection with the primitive will trigger one and only one invocation of the anyhit program. Otherwise, the anyhit program may be invoked more than once.
			DisableTriangleFaceCulling		= 1u << 2,		//	Prevent triangles from getting culled due to their orientation. Effectively ignores ray flags OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES and OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES.
		};

		//	Function to retrieve the subtype of the acceleration structure, indicating it as a geometry type.
		virtual SubType subType() const final { return SubType::Geometry; }

		//	Virtual function to retrieve the primitive type of the acceleration structure.
		virtual PrimitiveType primitiveType() const = 0;
	};

	/*****************************************************************************
	*************************    AccelStructTriangle    **************************
	*****************************************************************************/

	class AccelStructTriangle : public GeomAccelStruct
	{

	public:

		//	Build input for GAS with triangle primitive type.
		struct BuildInput
		{
			ns::BufferView1D<const ns::int3_16a>		indexBuffer = nullptr;				//	Optional pointer to array of int triplets, one triplet per triangle.
			ns::BufferView1D<const ns::float3_16a>		vertexBuffer = nullptr;				//	Pointer to array of positons on device memory. 
			ns::BufferView1D<const uint32_t>			sbtIndexOffsetBuffer = nullptr;		//	Device pointer to per-primitive local sbt index offset buffer. May be nullptr.
			ns::ArrayProxy<GeomFlags>					perSbtRecordFlags = nullptr;		//	Array of flags, size must match numSbtRecords. Passing nullptr will fill with GeomFlags::eNone.
			unsigned int								numIndexTriplets = 0;				//	Size of array in indexBuffer. If zeros, numIndexTriplets = numVertices / 3.
			unsigned int								numSbtRecords = 1;					//	Number of sbt records available to the sbt index offset override.
			unsigned int								numVertices = 0;					//	Number of vertices in each of buffer in vertexBuffer.
		};

		//	Returns a constant reference to a vector of BuildInput structures,  
		virtual const std::vector<BuildInput> & buildInputs() const = 0;

		//	Function to retrieve the primitive type of the acceleration structure, indicating it as a triangle primitive.
		virtual PrimitiveType primitiveType() const final { return PrimitiveType::Triangle; }

		//	Abstract function to build the acceleration structure from input triangles.
		virtual void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<BuildInput> buildInputs, bool preferFastTrace, bool allowUpdate) = 0;
	};

	/*****************************************************************************
	***************************    AccelStructAabb    ****************************
	*****************************************************************************/

	class AccelStructAabb : public GeomAccelStruct
	{

	public:

		//	Build input for GAS with aabb primitive type.
		struct BuildInput
		{
			ns::BufferView1D<const Aabb>			aabbBuffer = nullptr;				//	Pointer to AABBs on device memory.
			ns::BufferView1D<const uint32_t>		sbtIndexOffsetBuffer = nullptr;		//	Device pointer to per-primitive local sbt index offset buffer. May be nullptr.
			ns::ArrayProxy<GeomFlags>				perSbtRecordFlags = nullptr;		//	Array of flags, size must match numSbtRecords. Passing nullptr will fill with GeomFlags::eNone.
			unsigned int							numSbtRecords = 1;					//	Number of sbt records available to the sbt index offset override.
			unsigned int							numPrimitives = 0;					//	Number of primitives.
		};

		//	Returns a constant reference to a vector of BuildInput structures,  
		virtual const std::vector<BuildInput> & buildInputs() const = 0;

		//	Function to retrieve the primitive type of the acceleration structure, indicating it as a AABB primitive.
		virtual PrimitiveType primitiveType() const final { return PrimitiveType::AABB; }

		//	Abstract function to build the acceleration structure from input AABBs.
		virtual void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<BuildInput> buildInputs, bool preferFastTrace, bool allowUpdate) = 0;
	};

	/*****************************************************************************
	***************************    AccelStructCurve    ***************************
	*****************************************************************************/

	class AccelStructCurve : public GeomAccelStruct
	{

	public:

		//	Enum defining different curve types.
		enum CurveType
		{
			RoundLinear				= 0x2503,		//	Piecewise linear curve with circular cross-section.
			RoundCatmullRom			= 0x2504,		//	CatmullRom curve with circular cross-section.
			RoundCubicBezier		= 0x2507,		//	Bezier curve of degree 3 with circular cross-section.
			RoundCubicBSpline		= 0x2502,		//	B-spline curve of degree 3 with circular cross-section.
			FlatQuadraticBSpline	= 0x2505,		//	B-spline curve of degree 2 with oriented, flat cross-section.
			RoundQuadraticBSpline	= 0x2501,		//	B-spline curve of degree 2 with circular cross-section.
		};

		//	Build input for GAS with curve primitive type.
		struct BuildInput
		{
			CurveType									curveType = RoundLinear;			
			ns::BufferView1D<const ns::float3_16a>		vertexBuffer = nullptr;			//	Pointer to array of positons on device memory.
			ns::BufferView1D<const uint32_t>			indexBuffer = nullptr;			//	These define a single segment. Size of array is numPrimitives.
			ns::BufferView1D<const float>				widthBuffer = nullptr;			//	Specifying the curve width (radius) corresponding to each vertex.
			unsigned int								numPrimitives = 0;				//	Number of primitives.
			unsigned int								numVertices = 0;				//	Number of vertices in each buffer in vertexBuffers.
			GeomFlags									flags = None;					//	Combination of GeomFlags describing the primitive behavior.
		};

		//	Returns a constant reference to a vector of BuildInput structures,  
		virtual const std::vector<BuildInput> & buildInputs() const = 0;

		//	Function to retrieve the primitive type of the acceleration structure, indicating it as a curve primitive.
		virtual PrimitiveType primitiveType() const final { return PrimitiveType::Curve; }

		//	Abstract function to build the acceleration structure from input curves.
		virtual void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<BuildInput> buildInputs, bool preferFastTrace, bool allowUpdate) = 0;
	};

	/*****************************************************************************
	**************************    AccelStructSphere    ***************************
	*****************************************************************************/

	class AccelStructSphere : public GeomAccelStruct
	{

	public:

		//	Build input for GAS with sphere primitive type.
		struct BuildInput
		{
			ns::BufferView1D<const float>				radiusBuffer = nullptr;				//	Parallel to vertexBuffer: specifying the sphere radius corresponding to each vertex.
			ns::BufferView1D<const ns::float3_16a>		vertexBuffer = nullptr;				//	Pointer to array of positons on device memory.
			ns::BufferView1D<const uint32_t>			sbtIndexOffsetBuffer = nullptr;		//	Device pointer to per-primitive local sbt index offset buffer. May be nullptr.
			ns::ArrayProxy<GeomFlags>					perSbtRecordFlags = nullptr;		//	Array of flags, size must match numSbtRecords. Passing nullptr will fill with GeomFlags::eNone.
			unsigned int								numSbtRecords = 1;					//	Number of sbt records available to the sbt index offset override.
			unsigned int								numVertices = 0;					//	Number of vertices in each buffer in vertexBuffers.
			bool										singleRadius = false;				//	Boolean value indicating whether a single radius per radius buffer is used, or the number of radii in radiusBuffers equals numVertices.
		};

		//	Returns a constant reference to a vector of BuildInput structures,  
		virtual const std::vector<BuildInput> & buildInputs() const = 0;

		//	Function to retrieve the primitive type of the acceleration structure, indicating it as a sphere primitive.
		virtual PrimitiveType primitiveType() const final { return PrimitiveType::Sphere; }

		//	Abstract function to build the acceleration structure from input spheres.
		virtual void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<BuildInput> buildInputs, bool preferFastTrace, bool allowUpdate) = 0;
	};

	/*****************************************************************************
	***************************    InstAccelStruct    ****************************
	*****************************************************************************/

	class InstAccelStruct : virtual public AccelStruct
	{

	public:

		//	Flags set on the InstDesc::flags. These can be or'ed together to combine multiple flags.
		enum InstFlags : unsigned int
		{
			None								= 0,			//	No special flag set.
			DisableTriangleFaceCulling			= 1u << 0,		//	Prevent triangles from getting culled due to their orientation.
			eFlagFlipTriangleFacing				= 1u << 1,		//	Flip triangle orientation. This affects front/backface culling as well as the reported face in case of a hit.
			DisableAnyhit						= 1u << 2,		//	Disable anyhit programs for all geometries of the instance. Can be overridden by Flags::EnforceAnyhit. This flag is mutually exclusive with Flags::EnforceAnyhit.
			EnforceAnyhit						= 1u << 3,		//	Enables anyhit programs for all geometries of the instance. Overrides OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT. Can be overridden by Flags::eDisableAnyhit.
			ForceOpacityMicromapAsTwoState		= 1u << 4,		//	Force 4-state opacity micromaps to behave as 2-state opacity micromaps during traversal.
			DisableOpacityMicromaps				= 1u << 5,		//	Don't perform opacity micromap query for this instance. GAS must be built with ALLOW_DISABLE_OPACITY_MICROMAPS for this to be valid.
		};

		//	Build input for IAS.
		struct BuildInput
		{
			std::shared_ptr<GeomAccelStruct>	geomAccelStruct = nullptr;		//	Set with an OptixTraversableHandle.
			ns::BufferView1D<const Mat4x4>		transform = nullptr;			//	Pointer to the affine object-to-world transformation matrix in row-major layout.
			unsigned int						visibilityMask = 255;			//	Visibility mask. If rayMask & instanceMask == 0 the instance is culled.
			unsigned int						instanceId = 0;					//	Application supplied ID. The maximal ID can be queried using OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID.
			unsigned int						sbtOffset = 0;					//	SBT record offset. In a traversable graph with multiple levels of IAS objects, offsets are summed together.
			InstFlags							flags = None;					//	Any combination of OptixInstanceFlags is allowed.
		};

		//	Returns a constant reference to a vector of BuildInput structures,  
		virtual const std::vector<BuildInput> & buildInputs() const = 0;

		//	Function to retrieve the subtype of the acceleration structure, indicating it as a instance type.
		virtual SubType subType() const final { return SubType::Instance; }

		//	Abstract function to build the acceleration structure from input instances.
		virtual void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<BuildInput> buildInputs, bool preferFastTrace, bool allowUpdate) = 0;
	};
}