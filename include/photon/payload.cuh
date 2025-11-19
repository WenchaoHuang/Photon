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
#include <cuda_runtime.h>

#ifndef __CUDACC__
	#error This file should be inclued in *.cu files only!
#endif

namespace PHOTON_NAMESPACE
{
	/*****************************************************************************
	*******************************    Payload    ********************************
	*****************************************************************************/

	/**
	 *	@brief		A template struct to encapsulate a payload value with its corresponding indices for OptiX.
	 *	@details	This struct uses a union to map the payload value to an array of 32-bit unsigned integers,
	 *				allowing it to be passed to Optix's payload system. The number of slots is computed based
	 *				on the size of Type, rounded up to the nearest 32-bit boundary.
	 */
	template<typename Type, unsigned int... Indices> struct Payload
	{
		using value_type = Type;

		static constexpr int numSlots = (sizeof(Type) + 3) / 4;

		static_assert(sizeof...(Indices) == numSlots, "Number of indices must match the size of Type in 32-bit slots");

		union
		{
			struct {  value_type        value;       };
			struct { unsigned int encodes[numSlots]; };
		};

		__device__ Payload() {}
		__device__ Payload(value_type v) : value(v) {}
		__device__ operator value_type&() { return value; }
		__device__ operator const value_type&() const { return value; }
	};

	/*****************************************************************************
	******************************    IsPayload    *******************************
	*****************************************************************************/

	//	@brief		Type trait to detect whether a type is a `Payload`.
	template<typename Type>							 struct IsPayload							 { static constexpr bool value = false; };
	template<typename Type, unsigned int... Indices> struct IsPayload<Payload<Type, Indices...>> { static constexpr bool value = true;  };

	/*****************************************************************************
	*****************************    set_payload    ******************************
	*****************************************************************************/

	/**
	 *	@brief		Writes the 32-bit payload at the given slot index.
	 *	@details	There are up to 32 attributes available.
	 *	@example	set_payload<0>(count);
	 *	@see		`optixSetPayload_0()`, `get_payload<Index>()`.
	 *	@note		Available in IS, AH, CH, MS.
	 */
	template<unsigned int Index> requires(Index < 32) __device__ __forceinline__ void set_payload(unsigned int payload)
	{
		asm volatile("call _optix_set_payload, (%0, %1);" : : "r"(Index), "r"(payload) : );
	}


	/**
	 *	@brief		Sets multiple payload slots based on a Payload struct.
	 *	@details	This function unpacks the Payload's encodes array and sets each 32-bit value to the corresponding slot index using a fold expression.
	 *	@see		`set_payload<Index>(...)`
	 *	@example	Payload<float, 0> flags = 0;	set_payload(flags);
	 */
	template<typename Type, unsigned int... Indices> __device__ __forceinline__ void set_payload(Payload<Type, Indices...> payload)
	{
		int index = 0;			(set_payload<Indices>(payload.encodes[index++]), ...);
	}


	/**
	 *	@brief		Writes a value directly into multiple payload slots.
	 *	@details	This function constructs a temporary `Payload<Type, Indices...>` from the given value,
	 *				and forwards it to the multi-slot `set_payload(Payload<...>)` overload.
	 *	@example	set_payload<ns::float3, 0, 1, 2>(rayOrigin);
	 *	@warning	`Type` must not be a `Payload` for avoid ambiguity.
	 */
	template<typename Type, unsigned int... Indices> __device__ __forceinline__ void set_payload(Type value)
	{
		static_assert(!IsPayload<Type>::value, "Type must not be a Payload");

		set_payload<Type, Indices...>(Payload<Type, Indices...>(value));
	}

	/*****************************************************************************
	*****************************    get_payload    ******************************
	*****************************************************************************/

	/**
	 *	@brief		Returns the 32-bit payload at the given slot index.
	 *	@details	There are up to 32 attributes available.
	 *	@example	unsigned int count = get_payload<0>();
	 *	@see		`optixGetPayload_0()`, `set_payload<Index>()`. 
	 *	@note		Available in IS, AH, CH, MS.
	 */
	template<unsigned int Index> requires(Index < 32) __device__ __forceinline__ unsigned int get_payload()
	{
		unsigned int payload;		asm volatile("call (%0), _optix_get_payload, (%1);" : "=r"(payload) : "r"(Index) : );		return payload;
	}

	namespace details
	{
		//!	Primary implementation of payload retrieval.
		template<typename Type, unsigned int... Indices> struct GetPayloadImpl
		{
			static __device__ __forceinline__ Payload<Type, Indices...> invoke()
			{
				int index = 0;			Payload<Type, Indices...> payload;

				((payload.encodes[index++] = get_payload<Indices>()), ...);

				return payload;
			}
		};

		//!	Specialization of `GetPayloadImpl` for `Payload` types.
		template<typename Type, unsigned int... Indices> struct GetPayloadImpl<Payload<Type, Indices...>>
		{
			static __device__ __forceinline__ Payload<Type, Indices...> invoke()
			{
				return GetPayloadImpl<Type, Indices...>::invoke();
			}
		};
	}

	/**
	 *	@brief		Retrieves a payload value of `Type` from one or more slots.
	 *	@details	This function reads 32-bit values from the specified indices and assembles them into a single value of `Type` using a fold expression.
	 *	@example	auto rayOrigin = get_payload<ns::float3, 0, 1, 2>();
	 */
	template<typename Type, unsigned int... Indices> __device__ __forceinline__ auto get_payload()
	{
		return details::GetPayloadImpl<Type, Indices...>::invoke();
	}


	/**
	 *	@brief		Retrieves a payload value when `Type` is already a `Payload`.
	 *	@example	PayloadType value = get_payload<PayloadType>();
	 */
	template<typename Type> __device__ __forceinline__ Type get_payload()
	{
		static_assert(IsPayload<Type>::value, "Type must be a Payload");

		return details::GetPayloadImpl<Type>::invoke();
	}

	/*****************************************************************************
	***************************    __PayloadExample    ***************************
	*****************************************************************************/

	/**
	 *	@brief		Example usage of the `Payload` wrapper with OptiX programs.
	 */
	__device__ __inline__ void __PayloadExample()
	{
		/**
		 *	Step-1:	Define type aliases using the `Payload<Type, Indices...>` template:
		 *			- `SeedType` for random seed (`int`) stored in slot 0.
		 *			- `RayDirType` for ray direction (`float3`) stored in slots 1¨C3.
		 *			- `RayOrginType` for ray origin (`float3`) stored in slots 4¨C6.
		 */
		using SeedType = Payload<int, 0>;
		using RayDirType = Payload<float3, 1, 2, 3>;
		using RayOrginType = Payload<float3, 4, 5, 6>;

		
		/**
		 *	Step-2:	In `__raygen__`:
		 *			- Construct payload values for ray direction and ray origin.
		 *			- Forward them to OptiX using `set_payload(...)`.
		 */
		RayDirType rayDir;
		RayOrginType rayOrigin;
		set_payload(rayOrigin);
		set_payload(rayDir);


		/**
		 *	Step-3:	In `__closesthit__`:
		 *			- Retrieve the stored payload values using `get_payload<Type>()`.
		 */
		rayDir = get_payload<RayDirType>();
		rayOrigin = get_payload<RayOrginType>();
		rayOrigin = get_payload<float3, 4, 5, 6>();
	}
}