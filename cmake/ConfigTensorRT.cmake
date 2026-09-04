# Set TensorRT root directory
# You can override this by setting TRT_ROOT environment variable
# or passing -DTRT_ROOT=/path/to/tensorrt or -DTensorRT_ROOT=/path/to/tensorrt to CMake
include("${CMAKE_CURRENT_LIST_DIR}/ConfigDependencyDefaults.cmake")

set(_inferrt_tensorrt_using_default_root OFF)

inferrt_dependency_resolve_path(
    _inferrt_tensorrt_root _inferrt_tensorrt_origin tensorrt
    VARIABLES TRT_ROOT TensorRT_ROOT
    ENVIRONMENT_VARIABLES TRT_ROOT TensorRT_ROOT TENSORRT_ROOT
    PREFIX_PATHS ${CMAKE_PREFIX_PATH}
    REQUIRED_FILES include/NvInfer.h
)
if(_inferrt_tensorrt_root)
    inferrt_dependency_cache_set(
        TRT_ROOT "${_inferrt_tensorrt_root}" PATH
        "TensorRT installation directory"
        INFERRT_DEPENDENCY_TRT_ROOT "${_inferrt_tensorrt_origin}")
    if(_inferrt_tensorrt_origin STREQUAL "project-default")
        set(_inferrt_tensorrt_using_default_root ON)
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
unset(_inferrt_tensorrt_root)
unset(_inferrt_tensorrt_origin)
unset(INFERRT_TENSORRT_DEFAULT_ROOT)
