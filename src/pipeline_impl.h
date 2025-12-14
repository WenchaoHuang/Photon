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

#include "pipeline.h"
#include <optix.h>
#include <map>

namespace PHOTON_NAMESPACE
{
	class ModuleImpl;
	class ProgramImpl;

	/*****************************************************************************
	******************************    ModuleImpl    ******************************
	*****************************************************************************/

	class ModuleImpl : public Module, public std::enable_shared_from_this<ModuleImpl>
	{

	public:

		ModuleImpl(std::shared_ptr<DeviceContext> deviceContext, OptixModule hModule);

		~ModuleImpl();

	public:

		virtual std::shared_ptr<Program> at(const std::string & funcName) override;

		std::shared_ptr<DeviceContext> deviceContext() const { return m_deviceContext; }

	private:

		std::map<std::string, std::weak_ptr<ProgramImpl>>		m_programMap;

		const std::shared_ptr<DeviceContext>					m_deviceContext;

		const OptixModule										m_hModule;
	};

	/*****************************************************************************
	*****************************    ProgramImpl    ******************************
	*****************************************************************************/

	class ProgramImpl : public Program
	{

	public:

		ProgramImpl(std::shared_ptr<ModuleImpl> module, OptixProgramGroup hProgramGroup, Program::Type type);

		~ProgramImpl();

	public:

		virtual Type type() const override { return m_progType; }

		virtual const SbtHeader & header() const override { return m_header; }

		static Program::Type queryProgramType(const std::string & funcName);

		std::shared_ptr<DeviceContext> deviceContext() const;

		OptixProgramGroup handle() { return m_hProgramGroup; }

	private:

		const std::shared_ptr<ModuleImpl>		m_module;

		const OptixProgramGroup					m_hProgramGroup;

		const Program::Type						m_progType;

		SbtHeader								m_header;
	};
}