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

#include "denoiser.h"
#include "device_context.h"
#include <nucleus/array_1d.h>
#include <nucleus/array_2d.h>

namespace PHOTON_NAMESPACE
{
	/*****************************************************************************
	*****************************    DenoiserImpl    *****************************
	*****************************************************************************/

	class DenoiserImpl : public Denoiser
	{

	public:

		DenoiserImpl(std::shared_ptr<DeviceContext> deviceContext);

		virtual ~DenoiserImpl();

	public:

		virtual void release() override;
		virtual ModelKind modelKind() const override { return m_eModelKind; }
		virtual unsigned int maxInputWidth() const override { return m_maxInputWidth; }
		virtual unsigned int maxInputHeight() const override { return m_maxInputHeight; }
		virtual std::shared_ptr<class DeviceContext> deviceContext() override { return m_deviceContext; }
		virtual void nextFrame() override { m_internalGuideLayers[0].swap(m_internalGuideLayers[1]); }
		virtual void preallocate(ns::AllocPtr pAlloc, ModelKind eModeKind, unsigned int maxInputWidth, unsigned int maxInputHeight) override;
		virtual void launch(ns::Stream & stream, dev::Ptr2<Color4f> output, dev::Ptr2<const Color4f> input, dev::Ptr2<const Color4f> albedo, dev::Ptr2<const Color4f> normal,
							dev::Ptr2<const Color4f> previousOutput, dev::Ptr2<const ns::float2> flow, dev::Ptr2<const float> flowTrustworthiness, float blendFactor) override;

	protected:

		void internalSetup(ns::Stream & stream, unsigned int inputWidth, unsigned int inputHeight);

	protected:

		ModelKind									m_eModelKind;
		unsigned int								m_inputWidth;
		unsigned int								m_inputHeight;
		unsigned int								m_maxInputWidth;
		unsigned int								m_maxInputHeight;
		unsigned int								m_currInternalGuideLayer;
		OptixDenoiser								m_hDenoiser;
		ns::Array<unsigned char>					m_stateCache;
		ns::Array<unsigned char>					m_scratchCache;
		ns::Array<unsigned char>					m_avgColorCache;
		ns::Array<unsigned char>					m_intensityCache;
		ns::Array2D<unsigned char>					m_internalGuideLayers[2];
		const std::shared_ptr<DeviceContext>		m_deviceContext;
	};
}