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
#include "sbt_record.h"
#include <string>

namespace PHOTON_NAMESPACE
{
	/*****************************************************************************
	*******************************    Program    ********************************
	*****************************************************************************/

	/**
	 *	@brief		Abstract interface for an OptiX program.
	 *	@note		Represents a single OptiX program entry, such as a ray-generation,
	 *				miss, hit, or callable program. Each program has a type and an
	 *				associated SBT (Shader Binding Table) header.
	 */
	class Program
	{

	public:

		/**
		 *	@brief	Supported OptiX program types.
		 *	@note	Each type corresponds to a specific OptiX program entry function prefix or grouping.
		 */
		enum Type
		{
			Miss,						//	prefix "__miss__"
			AnyHit,						//	prefix "__anyhit__"
			Raygen,						//	prefix "__raygen__"
			Exception,					//	prefix "__exception__"
			ClosestHit,					//	prefix "__closesthit__"
			Intersection,				//	prefix "__intersection__"
			DirectCallable,				//	prefix "__direct_callable__"
			ContinuationCallable,		//	prefix "__continuation_callable__"
			BuiltinIntersection,
			CallableGroup,
			HitGroup,
			Unknow,
		};

		//!	Virtual destructor.
		virtual ~Program() {}

		//!	Get the type of this program.
		virtual Type type() const = 0;

		//!	Get the SBT header for this program.
		virtual const SbtHeader & header() const = 0;

		//	TODO
		static std::shared_ptr<Program> combine(std::shared_ptr<Program> program0, std::shared_ptr<Program> program1);
		static std::shared_ptr<Program> combine(std::shared_ptr<Program> program0, std::shared_ptr<Program> program1, std::shared_ptr<Program> program2);
	};

	/*****************************************************************************
	********************************    Module    ********************************
	*****************************************************************************/

	/**
	 *	@brief		Abstract interface for an OptiX module.
	 *	@note		Represents a compiled OptiX module that contains one or more
	 *				program entry points. Programs can be retrieved by name.
	 */
	class Module
	{

	public:

		//!	Virtual destructor.
		virtual ~Module() {}


		/**
		 *	@brief		Retrieve a program by its function entry name.
		 *	@note		The function name must match one of the PTX entry points defined
		 *				in this module, such as "__raygen__xxx" or "__miss__yyy".
		 * @param[in]	funcName - The PTX function entry name.
		 * @return		A shared pointer to the corresponding Program.
		 */
		virtual std::shared_ptr<Program> at(const std::string & funcName) = 0;
	};
}