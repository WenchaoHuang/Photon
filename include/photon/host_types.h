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

#include <nucleus/array_proxy.h>

namespace photon
{
	/*****************************************************************************
	******************************    DeviceProp    ******************************
	*****************************************************************************/

	struct DeviceProp
	{
		unsigned int	version;							//!	The RT core version supported by the device (0 for no support, 10 for version 1.0).
		unsigned int	clusterAccel;						//!	Flag specifying support for cluster acceleration structure builds.
		unsigned int	maxSbtOffset;						//!	The maximum value for OptixInstance::sbtOffset.
		unsigned int	maxInstanceID;						//!	The maximum value for OptixInstance::instanceId.
		unsigned int	maxTraceDepth;						//!	Maximum value for OptixPipelineLinkOptions::maxTraceDepth.
		unsigned int	maxPrimitivesPerGAS;				//!	The maximum number of primitives (over all build inputs) as input to a single Geometry Acceleration Structure (GAS).
		unsigned int	maxInstancesPerIAS;					//!	The maximum number for the sum of the number of SBT records of all build inputs to a single Geometry Acceleration Structure (GAS).
		unsigned int	maxSbtRecordsPerGAS;				//!	The maximum number of instances that can be added to a single Instance Acceleration Structure (IAS).
		unsigned int	maxTraversableGraphDepth;			//!	Maximum value to pass into optixPipelineSetStackSize.
		unsigned int	numBitsInstanceVisiblityMask;		//!	The number of bits available for the OptixInstance::visibilityMask.

	#if (OPTIX_VERSION >= 80100)
		unsigned int	shaderExecutionReordering;			//!	Flag specifying capabilities of the optixReorder() device function.
	#endif
	#if (OPTIX_VERSION >= 90000)
		unsigned int	cooperativeVector;					//!	Flag specifying whether cooperative vector support is enabled for this device.
		unsigned int	maxClusterVertices;					//!	The maximum unique vertices per cluster in a cluster acceleration structure builds.
		unsigned int	maxClusterTriangles;				//!	The maximum triangles per cluster in a cluster acceleration structure builds.
		unsigned int	maxStructuredGridResolution;		//!	The maximum resolution per cluster in a structured cluster acceleration structure builds.
	#endif
	};
}