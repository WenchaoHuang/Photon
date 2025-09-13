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

#include "denoiser_impl.h"
#include <nucleus/Logger.h>
#include <nucleus/Stream.h>
#include <optix_stubs.h>

PHOTON_USING_NAMESPACE

/*********************************************************************************
*******************************    DenoiserImpl    *******************************
*********************************************************************************/

DenoiserImpl::DenoiserImpl(std::shared_ptr<DeviceContextImpl> deviceContext) : m_deviceContext(deviceContext), m_hDenoiser(nullptr),
	m_eModelKind(ModelKind::Normal), m_maxInputWidth(0), m_maxInputHeight(0), m_inputWidth(0), m_inputHeight(0)
{

}


void DenoiserImpl::launch(ns::Stream & stream, dev::Ptr2<Color4f> output, dev::Ptr2<const Color4f> input, dev::Ptr2<const Color4f> albedo, dev::Ptr2<const Color4f> normal,
						  dev::Ptr2<const Color4f> previousOutput, dev::Ptr2<const ns::float2> flow, [[maybe_unused]] dev::Ptr2<const float> flowTrustworthiness, float blendFactor)
{
	NS_ASSERT((albedo.width() == input.width()) && (albedo.height() == input.height()));
	NS_ASSERT((normal.width() == input.width()) && (normal.height() == input.height()));

	OptixDenoiserLayer									denoiserLayer = {};
	OptixDenoiserParams									denoiserParams = {};
	OptixDenoiserGuideLayer								denoiserGuideLayer = {};

#if OPTIX_VERSION >= 70700
	denoiserLayer.type									= OPTIX_DENOISER_AOV_TYPE_NONE;
#endif
	denoiserLayer.input.format							= OPTIX_PIXEL_FORMAT_FLOAT4;
	denoiserLayer.input.data							= (CUdeviceptr)input.data();
	denoiserLayer.input.width							= input.width();
	denoiserLayer.input.height							= input.height();
	denoiserLayer.input.rowStrideInBytes				= input.pitch();
	denoiserLayer.input.pixelStrideInBytes				= sizeof(Color4f);

	denoiserLayer.output.format							= OPTIX_PIXEL_FORMAT_FLOAT4;
	denoiserLayer.output.data							= (CUdeviceptr)output.data();
	denoiserLayer.output.width							= output.width();
	denoiserLayer.output.height							= output.height();
	denoiserLayer.output.rowStrideInBytes				= output.pitch();
	denoiserLayer.output.pixelStrideInBytes				= sizeof(Color4f);

	denoiserGuideLayer.albedo.format					= OPTIX_PIXEL_FORMAT_FLOAT3;
	denoiserGuideLayer.albedo.data						= (CUdeviceptr)albedo.data();
	denoiserGuideLayer.albedo.width						= albedo.width();
	denoiserGuideLayer.albedo.height					= albedo.height();
	denoiserGuideLayer.albedo.rowStrideInBytes			= albedo.pitch();
	denoiserGuideLayer.albedo.pixelStrideInBytes		= sizeof(Color4f);
	
	denoiserGuideLayer.normal.format					= OPTIX_PIXEL_FORMAT_FLOAT3;
	denoiserGuideLayer.normal.data						= (CUdeviceptr)normal.data();
	denoiserGuideLayer.normal.width						= normal.width();
	denoiserGuideLayer.normal.height					= normal.height();
	denoiserGuideLayer.normal.rowStrideInBytes			= normal.pitch();
	denoiserGuideLayer.normal.pixelStrideInBytes		= sizeof(Color4f);

	denoiserParams.blendFactor							= blendFactor;
	denoiserParams.hdrAverageColor						= (CUdeviceptr)m_avgColorCache.data();
	denoiserParams.hdrIntensity							= (CUdeviceptr)m_intensityCache.data();
	denoiserParams.temporalModeUsePreviousLayers		= (m_eModelKind & ModelKind::Temporal) && !previousOutput.empty() && (previousOutput != input);
#if OPTIX_VERSION <= 70700
	denoiserParams.denoiseAlpha							= OPTIX_DENOISER_ALPHA_MODE_COPY;
#endif

	if (m_eModelKind & ModelKind::Temporal)
	{
		NS_ASSERT((flow.width() == input.width()) && (flow.height() == input.height()));

		if (previousOutput.empty())													previousOutput = input;

		denoiserLayer.previousOutput.format											= OPTIX_PIXEL_FORMAT_FLOAT4;
		denoiserLayer.previousOutput.data											= (CUdeviceptr)previousOutput.data();
		denoiserLayer.previousOutput.width											= previousOutput.width();
		denoiserLayer.previousOutput.height											= previousOutput.height();
		denoiserLayer.previousOutput.rowStrideInBytes								= previousOutput.pitch();
		denoiserLayer.previousOutput.pixelStrideInBytes								= sizeof(Color4f);

		denoiserGuideLayer.flow.format												= OPTIX_PIXEL_FORMAT_FLOAT2;
		denoiserGuideLayer.flow.data												= (CUdeviceptr)flow.data();
		denoiserGuideLayer.flow.width												= flow.width();
		denoiserGuideLayer.flow.height												= flow.height();
		denoiserGuideLayer.flow.rowStrideInBytes									= flow.pitch();
		denoiserGuideLayer.flow.pixelStrideInBytes									= sizeof(ns::float2);
	#if OPTIX_VERSION >= 70700
		denoiserGuideLayer.flowTrustworthiness.format								= OPTIX_PIXEL_FORMAT_FLOAT1;
		denoiserGuideLayer.flowTrustworthiness.data									= (CUdeviceptr)flowTrustworthiness.data();
		denoiserGuideLayer.flowTrustworthiness.width								= flowTrustworthiness.width();
		denoiserGuideLayer.flowTrustworthiness.height								= flowTrustworthiness.height();
		denoiserGuideLayer.flowTrustworthiness.rowStrideInBytes						= flowTrustworthiness.pitch();
		denoiserGuideLayer.flowTrustworthiness.pixelStrideInBytes					= sizeof(float);
	#endif
		denoiserGuideLayer.previousOutputInternalGuideLayer.format					= OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER;
		denoiserGuideLayer.previousOutputInternalGuideLayer.data					= (CUdeviceptr)m_internalGuideLayers[0].data();
		denoiserGuideLayer.previousOutputInternalGuideLayer.width					= input.width();
		denoiserGuideLayer.previousOutputInternalGuideLayer.height					= input.height();
		denoiserGuideLayer.previousOutputInternalGuideLayer.rowStrideInBytes		= static_cast<uint32_t>(m_internalGuideLayers[0].pitch());
		denoiserGuideLayer.previousOutputInternalGuideLayer.pixelStrideInBytes		= static_cast<uint32_t>(m_internalGuideLayers[0].pitch() / input.width());

		denoiserGuideLayer.outputInternalGuideLayer.format							= OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER;
		denoiserGuideLayer.outputInternalGuideLayer.data							= (CUdeviceptr)m_internalGuideLayers[1].data();
		denoiserGuideLayer.outputInternalGuideLayer.width							= input.width();
		denoiserGuideLayer.outputInternalGuideLayer.height							= input.height();
		denoiserGuideLayer.outputInternalGuideLayer.rowStrideInBytes				= static_cast<uint32_t>(m_internalGuideLayers[1].pitch());
		denoiserGuideLayer.outputInternalGuideLayer.pixelStrideInBytes				= static_cast<uint32_t>(m_internalGuideLayers[1].pitch() / input.width());

		if (!denoiserParams.temporalModeUsePreviousLayers)
		{
			//!	@see	NVIDIA Optix 9.0 Programming Guide >> Chater: Functions and data structures for denoising.
			//!	@note	The previousOutputInternalGuideLayer image content must be set to zero for the first frame.
			stream.memsetZero(m_internalGuideLayers[0].data(), m_internalGuideLayers[0].bytes());
		}
	}

	this->internalSetup(stream, denoiserLayer.input.width, denoiserLayer.input.height);

	OptixResult eResult = optixDenoiserComputeIntensity(m_hDenoiser, stream.handle(), &denoiserLayer.input,
														(CUdeviceptr)m_intensityCache.data(), (CUdeviceptr)m_scratchCache.data(), m_scratchCache.bytes());

	if (eResult == OPTIX_SUCCESS)
	{
		eResult = optixDenoiserComputeAverageColor(m_hDenoiser, stream.handle(), &denoiserLayer.input,
												   (CUdeviceptr)m_avgColorCache.data(), (CUdeviceptr)m_scratchCache.data(), m_scratchCache.bytes());

		if (eResult == OPTIX_SUCCESS)
		{
			eResult = optixDenoiserInvoke(m_hDenoiser, stream.handle(), &denoiserParams, (CUdeviceptr)m_stateCache.data(), m_stateCache.bytes(),
										  &denoiserGuideLayer, &denoiserLayer, 1, 0, 0, (CUdeviceptr)m_scratchCache.data(), m_scratchCache.bytes());
		}
	}

	if (eResult != OPTIX_SUCCESS)
	{
		NS_ERROR_LOG("Failed to invoke Optix denoiser: %s.", optixGetErrorString(eResult));

		throw eResult;
	}
}


void DenoiserImpl::preallocate(ns::AllocPtr pAlloc, ModelKind eModeKind, unsigned int maxInputWidth, unsigned int maxInputHeight)
{
	OptixDenoiserModelKind							denoiserModelKind = OPTIX_DENOISER_MODEL_KIND_AOV;
	if (eModeKind == Normal)						{ denoiserModelKind = OPTIX_DENOISER_MODEL_KIND_AOV; }
	else if (eModeKind == Upscale2x)				{ denoiserModelKind = OPTIX_DENOISER_MODEL_KIND_UPSCALE2X; }
	else if (eModeKind == Temporal)					{ denoiserModelKind = OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV; }
	else if (eModeKind == TemporalUpscale2x)		{ denoiserModelKind = OPTIX_DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X; }
	else											{ NS_ASSERT(false); }

	if ((m_maxInputWidth != maxInputWidth) || (m_maxInputHeight != maxInputHeight) || (m_eModelKind != eModeKind))
	{
		this->release();

		OptixDenoiser						hDenoiser = nullptr;
		OptixDenoiserOptions				denoiserOptions = {};
	#if OPTIX_VERSION >= 80000
		denoiserOptions.denoiseAlpha		= OPTIX_DENOISER_ALPHA_MODE_COPY;
	#endif
		denoiserOptions.guideAlbedo			= 1;
		denoiserOptions.guideNormal			= 1;

		OptixResult eResult = optixDenoiserCreate(m_deviceContext->handle(), denoiserModelKind, &denoiserOptions, &hDenoiser);

		if (eResult == OPTIX_SUCCESS)
		{
			OptixDenoiserSizes cacheSizes = {};

			eResult = optixDenoiserComputeMemoryResources(hDenoiser, maxInputWidth, maxInputHeight, &cacheSizes);

			if (eResult != OPTIX_SUCCESS)
			{
				eResult = optixDenoiserDestroy(hDenoiser);
			}
			else
			{
				m_internalGuideLayers[0].resize(pAlloc, cacheSizes.internalGuideLayerPixelSizeInBytes * maxInputWidth, maxInputHeight);
				m_internalGuideLayers[1].resize(pAlloc, cacheSizes.internalGuideLayerPixelSizeInBytes * maxInputWidth, maxInputHeight);
				m_scratchCache.resize(pAlloc, cacheSizes.withoutOverlapScratchSizeInBytes);
				m_avgColorCache.resize(pAlloc, cacheSizes.computeAverageColorSizeInBytes);
				m_intensityCache.resize(pAlloc, cacheSizes.computeIntensitySizeInBytes);
				m_stateCache.resize(pAlloc, cacheSizes.stateSizeInBytes);
				m_maxInputHeight = maxInputHeight;
				m_maxInputWidth = maxInputWidth;
				m_eModelKind = eModeKind;
				m_hDenoiser = hDenoiser;
				m_inputHeight = 0;
				m_inputWidth = 0;
			}
		}

		if (eResult != OPTIX_SUCCESS)
		{
			NS_ERROR_LOG("%s.", optixGetErrorString(eResult));

			throw eResult;
		}
	}
}


void DenoiserImpl::internalSetup(ns::Stream & stream, unsigned int inputWidth, unsigned int inputHeight)
{
	NS_ASSERT((inputWidth <= m_maxInputWidth) && (inputHeight <= m_maxInputHeight));

	if ((m_hDenoiser != nullptr) && ((m_inputWidth != inputWidth) || (m_inputHeight != inputHeight)))
	{
		OptixResult eResult = optixDenoiserSetup(m_hDenoiser, stream.handle(), inputWidth, inputHeight, (CUdeviceptr)m_stateCache.data(),
												 m_stateCache.bytes(), (CUdeviceptr)m_scratchCache.data(), m_scratchCache.bytes());

		if (eResult != OPTIX_SUCCESS)
		{
			NS_ERROR_LOG("Failed to steup Optix denoiser: %s.", optixGetErrorString(eResult));

			throw eResult;
		}

		m_inputHeight = inputHeight;
		m_inputWidth = inputWidth;
	}
}


void DenoiserImpl::release()
{
	if (m_hDenoiser != nullptr)
	{
		OptixResult eResult = optixDenoiserDestroy(m_hDenoiser);

		NS_ERROR_LOG_IF(eResult != OPTIX_SUCCESS, "Failed to destroy Optix denoiser: %s.", optixGetErrorString(eResult));

		m_stateCache.clear();
		m_scratchCache.clear();
		m_avgColorCache.clear();
		m_intensityCache.clear();
		m_internalGuideLayers[0].clear();
		m_internalGuideLayers[1].clear();
		m_eModelKind = ModelKind::Normal;
		m_maxInputWidth = m_maxInputHeight = 0;
		m_inputWidth = m_inputHeight = 0;
		m_hDenoiser = nullptr;
	}
}


DenoiserImpl::~DenoiserImpl()
{
	this->release();
}