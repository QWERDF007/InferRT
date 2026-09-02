# Set TensorRT root directory
# You can override this by setting TRT_ROOT environment variable
# or passing -DTRT_ROOT=/path/to/tensorrt or -DTensorRT_ROOT=/path/to/tensorrt to CMake
if(NOT DEFINED TRT_ROOT)
    if(DEFINED ENV{TRT_ROOT})
        set(TRT_ROOT "$ENV{TRT_ROOT}" CACHE PATH "TensorRT installation directory")
    elseif(DEFINED TensorRT_ROOT)
        set(TRT_ROOT "${TensorRT_ROOT}" CACHE PATH "TensorRT installation directory")
    elseif(DEFINED ENV{TensorRT_ROOT})
        set(TRT_ROOT "$ENV{TensorRT_ROOT}" CACHE PATH "TensorRT installation directory")
    endif()
endif()

# Include FindTensorRT to locate and configure TensorRT
include(${CMAKE_CURRENT_LIST_DIR}/FindTensorRT.cmake)

if(TRT_ROOT)
    set(TRT_ROOT "${TRT_ROOT}" CACHE PATH "TensorRT installation directory")
endif()
