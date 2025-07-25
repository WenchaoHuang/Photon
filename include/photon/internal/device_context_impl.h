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

#include <optix.h>
#include "device_context.h"

namespace photon
{
	/*****************************************************************************
	******************************    DeviceProp    ******************************
	*****************************************************************************/

	class ContextImpl : public DeviceContext, public std::enable_shared_from_this<ContextImpl>
	{

	public:

		explicit ContextImpl(ns::Device * device, OptixDeviceContext hContext, const DeviceProp & devProp);

		virtual ~ContextImpl();

	public:

		OptixDeviceContext handle() { return m_hContext; }

		virtual ns::Device * device() const override { return m_device; }

		virtual const DeviceProp & properties() const override { return m_devProp; }

	private:

		ns::Device * const				m_device;
		const OptixDeviceContext		m_hContext;
		DeviceProp						m_devProp;
	};
}