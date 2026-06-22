# Set TensorRT root directory
# You can override this by setting TRT_ROOT environment variable
# or passing -DTensorRT_ROOT=/path/to/tensorrt to CMake
set(TRT_ROOT "/home/pc/workspace/TensorRT-10.16.1.11" CACHE PATH "TensorRT installation directory" FORCE)
set(TRT_VERSION "10.16.1.11" CACHE STRING "TensorRT version" FORCE)

# Include FindTensorRT to locate and configure TensorRT
include(${CMAKE_CURRENT_LIST_DIR}/FindTensorRT.cmake)