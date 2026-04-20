# Set TensorRT root directory
# You can override this by setting TensorRT_ROOT environment variable
# or passing -DTensorRT_ROOT=/path/to/tensorrt to CMake
if(NOT DEFINED TensorRT_ROOT)
  set(TRT_ROOT "D:/Software/dev/TensorRT-10.16.1.11")
  set(TRT_VERSION "10.16.1.11")
endif()

# Include FindTensorRT to locate and configure TensorRT
if(NOT TARGET TensorRT::TensorRT)
    include(${CMAKE_CURRENT_LIST_DIR}/FindTensorRT.cmake)
endif()