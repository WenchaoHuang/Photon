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
#include "build_inputs.h"
#include <nucleus/span.h>
#include <nucleus/array_1d.h>
#include <nucleus/device_span.h>
#include <nucleus/vector_types.h>
#include <optix.h>

//	Backward compatibility: OptixBuildInputAabbArray was renamed to
//	OptixBuildInputCustomPrimitiveArray in OptiX 7.1.
#if OPTIX_VERSION < 70100
	typedef OptixBuildInputAabbArray		OptixBuildInputCustomPrimitiveArray;
#endif

namespace PHOTON_NAMESPACE
{
	/*****************************************************************************
	*****************************    AccelStruct    ******************************
	*****************************************************************************/

	/**
	 *	@brief		Base class for acceleration structures.
	 * 
	 *	@details	Manages GPU buffer state and implements `build` / `rebuild` / `refit` logic
	 *				shared by all concrete subclasses.
	 *	
	 *	A note on compaction (`OPTIX_BUILD_FLAG_ALLOW_COMPACTION`):
	 *	Compaction would reduce GAS memory footprint by ~30-50% for static geometry,
	 *	but introduces significant API and implementation complexity:
	 *	- Requires a CPU-GPU sync after `optixAccelBuild` to read the compacted size,
	 *	  breaking the async build pipeline.
	 *	- Is mutually exclusive with `OPTIX_BUILD_FLAG_ALLOW_UPDATE`, meaning it
	 *	  cannot be used with refit workflows (skinned meshes, particles, etc.).
	 *	- Has no measurable effect on traversal performance.
	 *	- Adds an extra GPU buffer and doubles the number of code paths in `build` / `rebuild` / `refit`.
	 *	
	 *	For these reasons, compaction is intentionally omitted from this wrapper.
	 *	The single output buffer path keeps the implementation lean and the build
	 *	pipeline fully asynchronous.
	 */
	class AccelStruct
	{
		NS_NONCOPYABLE(AccelStruct)

	public:

		//	Construct with the device context that owns this structure.
		PHOTON_API explicit AccelStruct(std::shared_ptr<class DeviceContext> deviceContext);

	public:

		//	Struct encapsulating build options for acceleration structure construction.
		struct BuildOptions
		{
			size_t					headerSize;
			OptixBuildFlags			buildFlags;
			OptixMotionOptions		motionOptions;
		};

		//	Returns true if the acceleration structure has not been built yet.
		bool empty() const { return m_hTraversable == 0; }

		//!	Returns the size of the header region in bytes (as passed to build()).
		size_t headerSize() const { return m_headerSize; }

		//	Returns the native OptiX traversable handle.
		OptixTraversableHandle handle() const { return m_hTraversable; }

		//	Returns the device context associated with this acceleration structure.
		std::shared_ptr<class DeviceContext> deviceContext() const { return m_deviceContext; }

		//	Returns true if the acceleration structure was built with update support.
		bool allowUpdate() const { return (m_buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE) != 0; }

		/**
		 *	@brief		Device pointer to the header buffer of a GAS (Geometry Acceleration Structure).
		 *
		 *	@details	This returns a pointer to the metadata header associated with the GAS.
		 *				The header region is placed before the OptiX acceleration data and is designed
		 *				for user-defined information such as primitive offsets, material indices,
		 *				or any custom per-GAS data layout.
		 *
		 *	@details	Memory layout example (including alignment considerations):
		 *				|------ user header buffer ------|--------- GAS output buffer --------- ... |
		 *				       ^ returned pointer                   ^BVH and geometry data
		 *
		 *	@note		The size of the header region will be aligned up to `OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT`.
		 *				That means the actual offset to the GAS acceleration data may be larger than the
		 *				raw header struct size and should be computed accordingly.
		 *
		 *	@note		This header is **intended to be accessed and interpreted by the application**,
		 *				and its structure is defined by the user. It can be used directly on device-side
		 *				kernels without copying back to host.
		 *
		 *	@example	auto geoBuffer = reinterpret_cast<const GeoBuffer*>(optixGetGASPointerFromHandle(optixGetGASTraversableHandle()) - headerSizeInBytes);
		 *	@note		API `optixGetGASPointerFromHandle` requires `OPTIX_VERSION >= 8.1.0`.
		 *
		 */
		dev::Span<unsigned char> gasHeaderBuffer()
		{
			return (m_headerSize != 0) ? dev::Span<unsigned char>(m_outputBuffer.data(), m_headerSize) : dev::Span<unsigned char>(nullptr, 0);
		}

	protected:

		//	Internal build helper called by each concrete subclass build() method.
		PHOTON_API void build(ns::Stream & stream, ns::AllocPtr allocator, const std::vector<OptixBuildInput> & buildInputs, BuildOptions buildOptions);

		//	Rebuilds the acceleration structure in-place using the last build inputs.
		PHOTON_API void rebuild(ns::Stream & stream, const std::vector<OptixBuildInput> & buildInputs);

		/**
		 *	@brief		Refits the acceleration structure without changing topology.
		 *	@warning	Only the device pointers and/or their buffer content may be changed.
		 */
		PHOTON_API void refit(ns::Stream & stream, const std::vector<OptixBuildInput> & buildInputs);

		size_t											m_headerSize;
		ns::Array<ns::byte>								m_tempBuffer;
		ns::Array<ns::byte>								m_outputBuffer;
		OptixTraversableHandle							m_hTraversable;
		OptixAccelBuildOptions							m_buildOptions;
		std::vector<OptixBuildInput>					m_cachedBuildInputs;
		const std::shared_ptr<class DeviceContext>		m_deviceContext;
	};

	/*****************************************************************************
	***************************    GeomAccelStruct    ****************************
	*****************************************************************************/

	//	Base class for geometry acceleration structures (GAS).
	class GeomAccelStruct : public AccelStruct
	{

	public:

		//	Construct with the device context that owns this GAS.
		explicit GeomAccelStruct(std::shared_ptr<class DeviceContext> deviceContext) : AccelStruct(std::move(deviceContext)) {}
	};

	/*****************************************************************************
	*************************    AccelStructTriangle    **************************
	*****************************************************************************/

	//!	@brief		Concrete GAS for built-in triangle primitives.
	class AccelStructTriangle : public GeomAccelStruct
	{

	public:

		//!	@brief	Constructs an empty triangle GAS associated with the given device context.
		explicit AccelStructTriangle(std::shared_ptr<class DeviceContext> deviceContext) : GeomAccelStruct(std::move(deviceContext)) {}

		//!	@brief	Builds the triangle GAS from the supplied build inputs.
		void build(ns::Stream & stream, ns::AllocPtr allocator, ns::Span<const BuildInputTriangles> buildInputs, BuildOptions buildOptions)
		{
			m_triangleBuildInputs.assign(buildInputs.begin(), buildInputs.end());

			AccelStruct::build(stream, allocator, this->prepareBuildInputs(), buildOptions);
		}

		//!	@brief	Rebuilds the triangle GAS from the supplied build inputs.
		void rebuild(ns::Stream & stream, ns::Span<const BuildInputTriangles> buildInputs = {})
		{
			if (!buildInputs.empty())
				m_triangleBuildInputs.assign(buildInputs.begin(), buildInputs.end());

			AccelStruct::rebuild(stream, this->prepareBuildInputs());
		}

		//!	@brief	Refits the triangle GAS from the supplied build inputs.
		void refit(ns::Stream & stream, ns::Span<const BuildInputTriangles> buildInputs = {})
		{
			if (!buildInputs.empty())
				m_triangleBuildInputs.assign(buildInputs.begin(), buildInputs.end());

			AccelStruct::refit(stream, this->prepareBuildInputs());
		}

	private:

		//!	@brief	Helper function to convert triangle build inputs to the generic `OptixBuildInput` format.
		const std::vector<OptixBuildInput> & prepareBuildInputs()
		{
			m_cachedBuildInputs.resize(m_triangleBuildInputs.size());

			for (size_t i = 0; i < m_cachedBuildInputs.size(); i++)
			{
				m_cachedBuildInputs[i] = {};
				m_cachedBuildInputs[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
				m_cachedBuildInputs[i].triangleArray = m_triangleBuildInputs[i].native();
			}
			return m_cachedBuildInputs;
		}

		std::vector<BuildInputTriangles>	m_triangleBuildInputs;
	};

	/*****************************************************************************
	***************************    AccelStructAabb    ****************************
	*****************************************************************************/

	//	Concrete GAS for custom AABB (user-defined intersection) primitives.
	class AccelStructAabb : public GeomAccelStruct
	{

	public:

		//!	@brief	Constructs an empty AABB GAS associated with the given device context.
		explicit AccelStructAabb(std::shared_ptr<class DeviceContext> deviceContext) : GeomAccelStruct(std::move(deviceContext)) {}

		//!	@brief	Builds the AABB GAS from the supplied build inputs.
		void build(ns::Stream & stream, ns::AllocPtr allocator, ns::Span<const BuildInputAabbs> buildInputs, BuildOptions buildOptions)
		{
			m_aabbBuildInputs.assign(buildInputs.begin(), buildInputs.end());

			AccelStruct::build(stream, allocator, this->prepareBuildInputs(), buildOptions);
		}

		//!	@brief	Rebuilds the AABB GAS from the supplied build inputs.
		void rebuild(ns::Stream & stream, ns::Span<const BuildInputAabbs> buildInputs = {})
		{
			if (!buildInputs.empty())
				m_aabbBuildInputs.assign(buildInputs.begin(), buildInputs.end());

			AccelStruct::rebuild(stream, this->prepareBuildInputs());
		}

		//!	@brief	Refits the AABB GAS from the supplied build inputs.
		void refit(ns::Stream & stream, ns::Span<const BuildInputAabbs> buildInputs = {})
		{
			if (!buildInputs.empty())
				m_aabbBuildInputs.assign(buildInputs.begin(), buildInputs.end());

			AccelStruct::refit(stream, this->prepareBuildInputs());
		}

	private:

		//!	@brief	Helper function to convert custom primitive build inputs to the generic `OptixBuildInput` format.
		const std::vector<OptixBuildInput> & prepareBuildInputs()
		{
			m_cachedBuildInputs.resize(m_aabbBuildInputs.size());

			for (size_t i = 0; i < m_cachedBuildInputs.size(); i++)
			{
				m_cachedBuildInputs[i] = {};
				m_cachedBuildInputs[i].type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
			#if OPTIX_VERSION >= 70100
				m_cachedBuildInputs[i].customPrimitiveArray = m_aabbBuildInputs[i].native();
			#else
				m_cachedBuildInputs[i].aabbArray = m_aabbBuildInputs[i].native();
			#endif
			}
			return m_cachedBuildInputs;
		}

		std::vector<BuildInputAabbs>	m_aabbBuildInputs;
	};

	/*****************************************************************************
	***************************    AccelStructCurve    ***************************
	*****************************************************************************/
	
#if OPTIX_VERSION >= 70100
	/**
	 *	@note		Requires Optix version >= 7.1.0
	 */
	//	Concrete GAS for built-in curve primitives.
	class AccelStructCurve : public GeomAccelStruct
	{

	public:

		//!	@brief	Constructs an empty curve GAS associated with the given device context.
		explicit AccelStructCurve(std::shared_ptr<class DeviceContext> deviceContext) : GeomAccelStruct(std::move(deviceContext)) {}

		//!	@brief	Builds the curve GAS from the supplied build inputs.
		void build(ns::Stream & stream, ns::AllocPtr allocator, ns::Span<const BuildInputCurves> buildInputs, BuildOptions buildOptions)
		{
			m_curveBuildInputs.assign(buildInputs.begin(), buildInputs.end());

			AccelStruct::build(stream, allocator, this->prepareBuildInputs(), buildOptions);
		}

		//!	@brief	Rebuilds the curve GAS from the supplied build inputs.
		void rebuild(ns::Stream & stream, ns::Span<const BuildInputCurves> buildInputs = {})
		{
			if (!buildInputs.empty())
				m_curveBuildInputs.assign(buildInputs.begin(), buildInputs.end());

			AccelStruct::rebuild(stream, this->prepareBuildInputs());
		}

		//!	@brief	Refits the curve GAS from the supplied build inputs.
		void refit(ns::Stream & stream, ns::Span<const BuildInputCurves> buildInputs = {})
		{
			if (!buildInputs.empty())
				m_curveBuildInputs.assign(buildInputs.begin(), buildInputs.end());

			AccelStruct::refit(stream, this->prepareBuildInputs());
		}

	private:

		//!	@brief	Helper function to convert curve build inputs to the generic `OptixBuildInput` format.
		const std::vector<OptixBuildInput> & prepareBuildInputs()
		{
			m_cachedBuildInputs.resize(m_curveBuildInputs.size());

			for (size_t i = 0; i < m_cachedBuildInputs.size(); i++)
			{
				m_cachedBuildInputs[i] = {};
				m_cachedBuildInputs[i].type = OPTIX_BUILD_INPUT_TYPE_CURVES;
				m_cachedBuildInputs[i].curveArray = m_curveBuildInputs[i].native();
			}
			return m_cachedBuildInputs;
		}

		std::vector<BuildInputCurves>		m_curveBuildInputs;
	};
#endif

	/*****************************************************************************
	**************************    AccelStructSphere    ***************************
	*****************************************************************************/

#if OPTIX_VERSION >= 70500
	/**
	 *	@note		Requires Optix version >= 7.5.0
	 */
	//	Concrete GAS for built-in sphere primitives.
	class AccelStructSphere : public GeomAccelStruct
	{

	public:

		//!	@brief	Constructs an empty sphere GAS associated with the given device context.
		explicit AccelStructSphere(std::shared_ptr<class DeviceContext> deviceContext) : GeomAccelStruct(std::move(deviceContext)) {}

		//!	@brief	Builds the sphere GAS from the supplied build inputs.
		void build(ns::Stream & stream, ns::AllocPtr allocator, ns::Span<const BuildInputSpheres> buildInputs, BuildOptions buildOptions)
		{
			m_sphereBuildInputs.assign(buildInputs.begin(), buildInputs.end());

			AccelStruct::build(stream, allocator, this->prepareBuildInputs(), buildOptions);
		}

		//!	@brief	Rebuilds the sphere GAS from the supplied build inputs.
		void rebuild(ns::Stream & stream, ns::Span<const BuildInputSpheres> buildInputs = {})
		{
			if (!buildInputs.empty())
				m_sphereBuildInputs.assign(buildInputs.begin(), buildInputs.end());

			AccelStruct::rebuild(stream, this->prepareBuildInputs());
		}

		//!	@brief	Refits the sphere GAS from the supplied build inputs.
		void refit(ns::Stream & stream, ns::Span<const BuildInputSpheres> buildInputs = {})
		{
			if (!buildInputs.empty())
				m_sphereBuildInputs.assign(buildInputs.begin(), buildInputs.end());

			AccelStruct::refit(stream, this->prepareBuildInputs());
		}

	private:

		//!	@brief	Helper function to convert sphere build inputs to the generic `OptixBuildInput` format.
		const std::vector<OptixBuildInput> & prepareBuildInputs()
		{
			m_cachedBuildInputs.resize(m_sphereBuildInputs.size());

			for (size_t i = 0; i < m_cachedBuildInputs.size(); i++)
			{
				m_cachedBuildInputs[i] = {};
				m_cachedBuildInputs[i].type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
				m_cachedBuildInputs[i].sphereArray = m_sphereBuildInputs[i].native();
			}
			return m_cachedBuildInputs;
		}

		std::vector<BuildInputSpheres>		m_sphereBuildInputs;
	};
#endif

	/*****************************************************************************
	***************************    InstAccelStruct    ****************************
	*****************************************************************************/

	//	Concrete IAS (Instance Acceleration Structure).
	class InstAccelStruct : public AccelStruct
	{

	public:

		//!	@brief	Constructs an empty IAS associated with the given device context.
		explicit InstAccelStruct(std::shared_ptr<class DeviceContext> deviceContext) : AccelStruct(std::move(deviceContext)) {}

		//!	@brief	Uploads the current host-side instance data and then builds the IAS.
		void build(ns::Stream & stream, ns::AllocPtr allocator, dev::Span<const OptixInstance> instances, size_t numInstances, BuildOptions buildOptions)
		{
			AccelStruct::build(stream, allocator, this->makeBuildInput(instances, numInstances), buildOptions);
		}

		//!	@brief	Uploads the current host-side instance data and then rebuilds the IAS.
		void rebuild(ns::Stream & stream, dev::Span<const OptixInstance> instances, size_t numInstances)
		{
			AccelStruct::rebuild(stream, this->makeBuildInput(instances, numInstances));
		}

		//!	@brief	Uploads the current host-side instance data and then refits the IAS.
		void refit(ns::Stream & stream, dev::Span<const OptixInstance> instances, size_t numInstances)
		{
			AccelStruct::refit(stream, this->makeBuildInput(instances, numInstances));
		}

	private:

		//!	@brief	Helper function to convert instance build input to the generic `OptixBuildInput` format.
		std::vector<OptixBuildInput> & makeBuildInput(dev::Span<const OptixInstance> instances, size_t numInstances)
		{
			m_cachedBuildInputs.resize(1);
			m_cachedBuildInputs[0].type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
			m_cachedBuildInputs[0].instanceArray.instances = (CUdeviceptr)instances.data();
			m_cachedBuildInputs[0].instanceArray.numInstances = static_cast<unsigned int>(numInstances);
			m_cachedBuildInputs[0].instanceArray.instanceStride = sizeof(OptixInstance);
			return m_cachedBuildInputs;
		}
	};
}
