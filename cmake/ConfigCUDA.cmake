# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if(NOT DEFINED INFERRT_CUDA_ROOT)
    if(DEFINED ENV{CUDA_PATH})
        set(INFERRT_CUDA_ROOT "$ENV{CUDA_PATH}" CACHE PATH "CUDA Toolkit root directory")
    elseif(DEFINED ENV{CUDA_TOOLKIT_ROOT_DIR})
        set(INFERRT_CUDA_ROOT "$ENV{CUDA_TOOLKIT_ROOT_DIR}" CACHE PATH "CUDA Toolkit root directory")
    endif()
endif()
if(INFERRT_CUDA_ROOT AND NOT DEFINED CUDAToolkit_ROOT)
    set(CUDAToolkit_ROOT "${INFERRT_CUDA_ROOT}" CACHE PATH "CUDA Toolkit root directory")
endif()

string(REPLACE "." ";" CUDA_VERSION_LIST ${CMAKE_CUDA_COMPILER_VERSION})
list(GET CUDA_VERSION_LIST 0 CUDA_VERSION_MAJOR)
list(GET CUDA_VERSION_LIST 1 CUDA_VERSION_MINOR)
list(GET CUDA_VERSION_LIST 2 CUDA_VERSION_PATCH)

find_package(CUDAToolkit ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR} REQUIRED)

# CMake 3.20 的 FindCUDAToolkit 尚未提供 CUDA::nvml，在这里统一补齐该 target。
if(NOT TARGET CUDA::nvml)
    find_library(INFERRT_NVML_LIBRARY
        NAMES nvml nvidia-ml
        HINTS ${CUDAToolkit_LIBRARY_DIR}
    )
    if(NOT INFERRT_NVML_LIBRARY)
        message(FATAL_ERROR "NVML library was not found in the CUDA toolkit")
    endif()
    add_library(CUDA::nvml UNKNOWN IMPORTED)
    set_target_properties(CUDA::nvml PROPERTIES
        IMPORTED_LOCATION "${INFERRT_NVML_LIBRARY}"
    )
    target_include_directories(CUDA::nvml SYSTEM INTERFACE ${CUDAToolkit_INCLUDE_DIRS})
endif()

# CUDA version requirement:
# - to use gcc-9 (11.4)

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11.8")
    message(FATAL_ERROR "Minimum CUDA version supported is 11.8")
endif()

# CUDA-specific options are applied per InferRT target by
# inferrt_apply_compile_options().
set(INFERRT_CUDA_TOOLKIT_OPTIONS -Xfatbin=--compress-all --extended-lambda)

# see https://developer.nvidia.com/cuda-gpus
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "$ENV{CUDAARCHS}")

    if(ARCH_X86_64)
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "13.0")
            list(APPEND CMAKE_CUDA_ARCHITECTURES
                70-real # Volta  - gv100/Tesla
            )
        endif()
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.8")
            list(APPEND CMAKE_CUDA_ARCHITECTURES
                75-real # Turing - tu10x/GeForce
                80-real # Ampere - ga100/Tesla
                86-real # Ampere - ga10x/GeForce
                89-real # Ada    - ad102/GeForce
                90-real # Hopper - gh100/Tesla
            )
        endif()
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8")
            list(APPEND CMAKE_CUDA_ARCHITECTURES
                100-real # Blackwell B200, B300
                120-real # RTX Pro 6000, RTX 50**
            )
        endif()
    elseif(ARCH_AARCH64)
        list(APPEND CMAKE_CUDA_ARCHITECTURES
            80-real # Ampere - ga100/Tesla
            86-real # Jetson IGX Orin with optional Ampere RTX A6000
            87-real # Ampere - ga10b,ga10c/Tegra (Jetson AGX Orin)
        )
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "13.0")
            list(APPEND CMAKE_CUDA_ARCHITECTURES
                72-real # Volta  - gv11b/Tegra (Jetson AGX Xavier)
            )
        endif()
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.8")
            list(APPEND CMAKE_CUDA_ARCHITECTURES
                89-real # Jetson IGX Orin with optional RTX 6000 Ada
                90-real # Grace Hopper - gh100/Tesla
            )
        endif()
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8")
            list(APPEND CMAKE_CUDA_ARCHITECTURES
                100-real # Blackwell GB200, GB300
            )
        endif()
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13.0")
            list(APPEND CMAKE_CUDA_ARCHITECTURES
                110-real # Thor
                121-real # DGX Spark
            )
        endif()
    endif()

    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "13.0")
        # Required compute capability:
        # * compute_70: fast fp16 support + PTX for forward compatibility
        list(APPEND CMAKE_CUDA_ARCHITECTURES 70-virtual)
    endif()

    # We must set the cache to the correct values, or else cmake will write its default there,
    # which is the old architecture supported by nvcc. We don't want that.
    set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures to build for")
endif()

find_library(CUDNN_LIB cudnn HINTS ${CUDAToolkit_LIBRARY_DIR})
