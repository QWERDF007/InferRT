# Set TensorRT root directory
# You can override this by setting TRT_ROOT environment variable
# or passing -DTensorRT_ROOT=/path/to/tensorrt to CMake
if(NOT DEFINED TRT_ROOT)
  set(TRT_ROOT "D:/Software/TensorRT-10.16.0.72" CACHE PATH "TensorRT installation directory")
  set(TRT_VERSION "10.16.0.72" CACHE STRING "TensorRT version")
endif()

# Include FindTensorRT to locate and configure TensorRT
if(NOT TARGET TensorRT::TensorRT)
    include(${CMAKE_CURRENT_LIST_DIR}/FindTensorRT.cmake)
endif()