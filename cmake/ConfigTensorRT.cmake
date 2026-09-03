# Set TensorRT root directory
# You can override this by setting TRT_ROOT environment variable
# or passing -DTRT_ROOT=/path/to/tensorrt or -DTensorRT_ROOT=/path/to/tensorrt to CMake
include("${CMAKE_CURRENT_LIST_DIR}/ConfigDependencyDefaults.cmake")

set(_inferrt_tensorrt_using_default_root OFF)

if(NOT DEFINED TRT_ROOT OR TRT_ROOT STREQUAL "")
    if(DEFINED TensorRT_ROOT AND NOT TensorRT_ROOT STREQUAL "")
        set(TRT_ROOT "${TensorRT_ROOT}" CACHE PATH "TensorRT installation directory")
    elseif(DEFINED ENV{TRT_ROOT} AND NOT "$ENV{TRT_ROOT}" STREQUAL "")
        set(TRT_ROOT "$ENV{TRT_ROOT}" CACHE PATH "TensorRT installation directory")
    elseif(DEFINED ENV{TensorRT_ROOT} AND NOT "$ENV{TensorRT_ROOT}" STREQUAL "")
        set(TRT_ROOT "$ENV{TensorRT_ROOT}" CACHE PATH "TensorRT installation directory")
    elseif(WIN32)
        inferrt_dependency_default(tensorrt _inferrt_tensorrt_default_root)
        if(_inferrt_tensorrt_default_root)
            set(INFERRT_TENSORRT_DEFAULT_ROOT "${_inferrt_tensorrt_default_root}")
            if(NOT CMAKE_PREFIX_PATH)
                set(TRT_ROOT "${_inferrt_tensorrt_default_root}" CACHE PATH
                    "TensorRT installation directory")
                set(_inferrt_tensorrt_using_default_root ON)
            endif()
        endif()
    endif()
endif()

if(_inferrt_tensorrt_using_default_root AND
   (NOT DEFINED TRT_VERSION OR TRT_VERSION STREQUAL ""))
    set(TRT_VERSION "10.16.1.11" CACHE STRING "TensorRT version")
endif()

# Include FindTensorRT to locate and configure TensorRT
include(${CMAKE_CURRENT_LIST_DIR}/FindTensorRT.cmake)

if(TRT_ROOT)
    set(TRT_ROOT "${TRT_ROOT}" CACHE PATH "TensorRT installation directory")
endif()

unset(_inferrt_tensorrt_using_default_root)
unset(_inferrt_tensorrt_default_root)
unset(INFERRT_TENSORRT_DEFAULT_ROOT)
