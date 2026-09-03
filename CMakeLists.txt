cmake_minimum_required(VERSION 3.25)

project("InferRT"
        LANGUAGES C CXX
        VERSION 0.0.2
        DESCRIPTION "推理相关工具"
)

# MSVC 下所有目标统一使用动态运行库，避免第三方静态库（如
# GoogleTest/GoogleMock）与测试目标在 /MT、/MD 之间产生链接冲突。
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

# 确保 cuda 主机编译器与我们使用的一致
if(NOT CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
endif()

# Optional suffix for non-release builds (for example, ``-beta``).
set(PROJECT_VERSION_SUFFIX "" CACHE STRING "Optional InferRT version suffix")

string(TOUPPER ${PROJECT_NAME} PROJECT_NAME_UPPER)
string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWER)

# 配置构建树的选项
option(${PROJECT_NAME_UPPER}_ENABLE_CUDA "Enable CUDA and GPU features" ON)
option(${PROJECT_NAME_UPPER}_BUILD_TENSORRT "Build TensorRT model/engine/features modules" ON)
option(${PROJECT_NAME_UPPER}_BUILD_TESTS "Enable testsuite" ON)
option(${PROJECT_NAME_UPPER}_BUILD_DOCS "Build Docs" OFF)
option(${PROJECT_NAME_UPPER}_BUILD_SAMPLES "Build Samples" ON)
option(${PROJECT_NAME_UPPER}_ENABLE_SANITIZER "Enabled sanitized build" OFF)
option(${PROJECT_NAME_UPPER}_BUILD_BENCHMARK "Build Benchmark" ON)
option(${PROJECT_NAME_UPPER}_BUILD_PYTHON "Build Python bindings" ON)
option(${PROJECT_NAME_UPPER}_BUILD_ONNX "Build the ONNX Runtime backend" OFF)
option(${PROJECT_NAME_UPPER}_BUILD_OPENVINO "Build the OpenVINO backend" OFF)

if(${PROJECT_NAME_UPPER}_BUILD_TENSORRT AND NOT ${PROJECT_NAME_UPPER}_ENABLE_CUDA)
    message(FATAL_ERROR "INFERRT_BUILD_TENSORRT requires INFERRT_ENABLE_CUDA=ON")
endif()
if(((${PROJECT_NAME_UPPER}_BUILD_ONNX) OR (${PROJECT_NAME_UPPER}_BUILD_OPENVINO))
   AND NOT ${PROJECT_NAME_UPPER}_BUILD_TENSORRT)
    message(FATAL_ERROR "ONNX/OpenVINO backends require INFERRT_BUILD_TENSORRT=ON")
endif()

if(${PROJECT_NAME_UPPER}_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
    else()
        message(FATAL_ERROR "CUDA language is requested but no CUDA compiler found.")
    endif()
endif()

# 配置构建树
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" )

include(ConfigVersion)
include(ConfigInstallPrefix)
include(ConfigBuildTree)
include(ConfigCompiler)
if(${PROJECT_NAME_UPPER}_ENABLE_CUDA)
    include(ConfigCUDA)
    include(ConfigOpenCV)
    if(${PROJECT_NAME_UPPER}_BUILD_TENSORRT)
        include(ConfigTensorRT)
        include(ConfigFaiss)
    endif()
endif()
if(${PROJECT_NAME_UPPER}_BUILD_PYTHON AND ${PROJECT_NAME_UPPER}_ENABLE_CUDA
   AND ${PROJECT_NAME_UPPER}_BUILD_TENSORRT)
    include(ConfigPython)
endif()
include(AddPluginLibrary)
include(AddSample)
include(AddTest)
include(AddBenchmark)
if(${PROJECT_NAME_UPPER}_BUILD_ONNX)
    include(ConfigONNXRuntime)
endif()
if(${PROJECT_NAME_UPPER}_BUILD_OPENVINO)
    include(ConfigOpenVINO)
endif()

# 第三方库
add_subdirectory(3rdparty EXCLUDE_FROM_ALL)

add_subdirectory(src)

# 构建测试用例
if(${PROJECT_NAME_UPPER}_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# 构建示例
if (${PROJECT_NAME_UPPER}_BUILD_SAMPLES)
    add_subdirectory(samples)
endif()

if (${PROJECT_NAME_UPPER}_BUILD_BENCHMARK)
    add_subdirectory(benchmark)
endif()

include(ConfigCMakePackage)

message(STATUS "")
include(PrintConfig)
