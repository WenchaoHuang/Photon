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

#include "accel_struct.h"
#include "device_context_impl.h"
#include <nucleus/array_1d.h>
#include <optix.h>

#pragma warning(disable: 4250)

namespace PHOTON_NAMESPACE
{
	class DeviceContextImpl;

	/*****************************************************************************
	***************************    AccelStructBase    ****************************
	*****************************************************************************/

	class AccelStructBase : virtual public AccelStruct
	{

	public:

		AccelStructBase(std::shared_ptr<DeviceContextImpl> deviceContext);

		~AccelStructBase();

	public:

		virtual bool empty() const override { return m_hTraversable == 0; }

		virtual std::shared_ptr<class DeviceContext> deviceContext() const override { return m_deviceContext; }

		virtual bool allowUpdate() const override { return (m_buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE) != 0; }

		virtual OptixTraversableHandle handle() const override { return m_hTraversable; }

		virtual void rebuild(ns::Stream & stream) override;

		virtual void refit(ns::Stream & stream) override;

	public:

		void build(ns::Stream & stream, ns::AllocPtr allocator, const std::vector<OptixBuildInput> & buildInputs, OptixAccelBuildOptions buildOptions, size_t headerSize);

		bool allowCompaction() const { return (m_buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0; }

		dev::Ptr<unsigned char> gasHeaderBuffer()
		{
			if ((m_headerSize != 0) && this->allowCompaction())
				return dev::Ptr<unsigned char>(m_compactedBuffer.data(), m_headerSize);
			else if (m_headerSize != 0)
				return dev::Ptr<unsigned char>(m_outputBuffer.data(), m_headerSize);
			else
				return dev::Ptr<unsigned char>(nullptr);
		}

	protected:

		size_t											m_headerSize;
		unsigned int									m_numSbtRecords;

	private:

		ns::Array<unsigned char>						m_tempBuffer;
		ns::Array<unsigned char>						m_outputBuffer;
		ns::Array<unsigned char>						m_compactedBuffer;
		OptixTraversableHandle							m_hTraversable;
		OptixAccelBuildOptions							m_buildOptions;
		std::vector<OptixBuildInput>					m_buildInputs;
		const std::shared_ptr<DeviceContextImpl>		m_deviceContext;
	};

	/*****************************************************************************
	***********************    AccelStructTriangleImpl    ************************
	*****************************************************************************/

	class AccelStructTriangleImpl : public AccelStructTriangle, public AccelStructBase
	{

	public:

		AccelStructTriangleImpl(std::shared_ptr<DeviceContextImpl> deviceContext) : AccelStructBase(deviceContext) {}

	public:

		virtual void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<BuildInput> buildInputs, size_t headerSize, bool preferFastTrace, bool allowUpdate) override;

		virtual const std::vector<BuildInput> & buildInputs() const override { return m_buildInputs; }

		virtual dev::Ptr<unsigned char> headerBuffer() override { return this->gasHeaderBuffer(); }

		virtual unsigned int numSbtRecords() const override { return m_numSbtRecords; }

		virtual size_t headerSize() const override { return m_headerSize; }

	private:

		std::vector<BuildInput>						m_buildInputs;
		std::vector<CUdeviceptr>					m_vertBuffers;
		std::vector<std::vector<unsigned int>>		m_geomFlags;
	};

	/*****************************************************************************
	*************************    AccelStructAabbImpl    **************************
	*****************************************************************************/

	class AccelStructAabbImpl : public AccelStructAabb, public AccelStructBase
	{

	public:

		AccelStructAabbImpl(std::shared_ptr<DeviceContextImpl> deviceContext) : AccelStructBase(deviceContext) {}

	public:

		virtual void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<BuildInput> buildInputs, size_t headerSize, bool preferFastTrace, bool allowUpdate) override;
		
		virtual const std::vector<BuildInput> & buildInputs() const override { return m_buildInputs; }

		virtual dev::Ptr<unsigned char> headerBuffer() override { return this->gasHeaderBuffer(); }

		virtual unsigned int numSbtRecords() const override { return m_numSbtRecords; }

		virtual size_t headerSize() const override { return m_headerSize; }

	private:

		std::vector<BuildInput>						m_buildInputs;
		std::vector<CUdeviceptr>					m_aabbBuffers;
		std::vector<std::vector<unsigned int>>		m_geomFlags;
	};

	/*****************************************************************************
	*************************    AccelStructCurveImpl    *************************
	*****************************************************************************/

	class AccelStructCurveImpl : public AccelStructCurve, public AccelStructBase
	{

	public:

		AccelStructCurveImpl(std::shared_ptr<DeviceContextImpl> deviceContext) : AccelStructBase(deviceContext) {}

	public:

		virtual void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<BuildInput> buildInputs, size_t headerSize, bool preferFastTrace, bool allowUpdate) override;

		virtual const std::vector<BuildInput> & buildInputs() const override { return m_buildInputs; }

		virtual dev::Ptr<unsigned char> headerBuffer() override { return this->gasHeaderBuffer(); }

		virtual unsigned int numSbtRecords() const override { return m_numSbtRecords; }

		virtual size_t headerSize() const override { return m_headerSize; }

	private:

		std::vector<BuildInput>			m_buildInputs;
		std::vector<CUdeviceptr>		m_vertBuffers;
		std::vector<CUdeviceptr>		m_widthBuffers;
	};

	/*****************************************************************************
	************************    AccelStructSphereImpl    *************************
	*****************************************************************************/

	class AccelStructSphereImpl : public AccelStructSphere, public AccelStructBase
	{

	public:

		AccelStructSphereImpl(std::shared_ptr<DeviceContextImpl> deviceContext) : AccelStructBase(deviceContext) {}

	public:

		virtual void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<BuildInput> buildInputs, size_t headerSize, bool preferFastTrace, bool allowUpdate) override;

		virtual const std::vector<BuildInput> & buildInputs() const override { return m_buildInputs; }

		virtual dev::Ptr<unsigned char> headerBuffer() override { return this->gasHeaderBuffer(); }

		virtual unsigned int numSbtRecords() const override { return m_numSbtRecords; }

		virtual size_t headerSize() const override { return m_headerSize; }

	private:

		std::vector<BuildInput>						m_buildInputs;
		std::vector<CUdeviceptr>					m_vertBuffers;
		std::vector<CUdeviceptr>					m_radiusBuffers;
		std::vector<std::vector<unsigned int>>		m_geomFlags;
	};

	/*****************************************************************************
	*************************    InstAccelStructImpl    **************************
	*****************************************************************************/

	class InstAccelStructImpl : public InstAccelStruct, public AccelStructBase
	{

	public:

		InstAccelStructImpl(std::shared_ptr<DeviceContextImpl> deviceContext) : AccelStructBase(deviceContext) {}

	public:

		virtual const std::vector<BuildInput> & buildInputs() const override { return m_buildInputs; }

		virtual void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<BuildInput> buildInputs, bool preferFastTrace, bool allowUpdate) override;

		virtual void rebuild(ns::Stream & stream) override;

		virtual void refit(ns::Stream & stream) override;

	private:

		std::vector<BuildInput>						m_buildInputs;
		ns::Array<ns::dev::Ptr<const Mat4x4>>		m_transforms;
		ns::Array<OptixInstance>					m_instances;
	};
}