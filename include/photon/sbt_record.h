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
#include <optix_types.h>

#pragma warning(disable: 4324)		//!	structure was padded due to alignment specifier.

namespace PHOTON_NAMESPACE
{
	/*****************************************************************************
	*****************************    SbtRecord<T>    *****************************
	*****************************************************************************/

	/**
	 *	@brief		Struct representing the header of a Shader Binding Table (SBT) record.
	 */
	struct SbtHeader { unsigned char storage[OPTIX_SBT_RECORD_HEADER_SIZE]; };


	/**
	 *	@brief		Template struct for creating Shader Binding Table (SBT) records with specified alignment and data type.
	 *	@tparam		SbtData - The type of data contained in the SBT record.
	 */
	template<typename SbtData> struct NS_ALIGN(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord
	{
		SbtHeader	header;		//	The header of the SBT record.
		SbtData		data;		//	The data contained in the SBT record.
	};


	/**
	 *	@brief		Specialization of SbtRecord struct for the case when SbtData is void.
	 */
	template<> struct NS_ALIGN(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord<void>
	{
		SbtHeader	header;		//	The header of the SBT record.
	};


	//	Type alias
	using EmptyRecord = SbtRecord<void>;
}