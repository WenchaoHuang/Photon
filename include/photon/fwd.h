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

#include "macros.h"
#include <nucleus/fwd.h>

/*********************************************************************************
***************************    Forward Declarations    ***************************
*********************************************************************************/

namespace PHOTON_NAMESPACE
{
	class Module;
	class Program;
	class Denoiser;
	class DeviceContext;

	class AccelStruct;
	class InstAccelStruct;
	class GeomAccelStruct;
	class AccelStructAabb;
	class AccelStructCurve;
	class AccelStructSphere;
	class AccelStructTriangle;

	struct NS_ALIGN(16) Color4f { float r, g, b, a; };
}

/*********************************************************************************
********************************    Type Alias    ********************************
*********************************************************************************/

using PtModule							= PHOTON_NAMESPACE::Module;
using PtDenoiser						= PHOTON_NAMESPACE::Denoiser;
using PtAccelStruct						= PHOTON_NAMESPACE::AccelStruct;
using PtDeviceContext					= PHOTON_NAMESPACE::DeviceContext;
using PtInstAccelStruct					= PHOTON_NAMESPACE::InstAccelStruct;
using PtGeomAccelStruct					= PHOTON_NAMESPACE::GeomAccelStruct;
using PtAccelStructAabb					= PHOTON_NAMESPACE::AccelStructAabb;
using PtAccelStructCurve				= PHOTON_NAMESPACE::AccelStructCurve;
using PtAccelStructSphere				= PHOTON_NAMESPACE::AccelStructSphere;
using PtAccelStructTriangle				= PHOTON_NAMESPACE::AccelStructTriangle;
