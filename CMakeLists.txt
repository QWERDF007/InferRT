cmake_minimum_required(VERSION 3.20)
include(CMakePackageConfigHelpers)

project("InferRT"
        LANGUAGES C CXX
        VERSION 0.0.1
        DESCRIPTION "推理相关工具"
)

# 确保 cuda 主机编译器与我们使用的一致，
if(NOT CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
endif()

enable_language(CUDA)

# Used when creating special builds
set(PROJECT_VERSION_SUFFIX "-beta")

# 如果用户没指定安装前缀
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/${PROJECT_NAME}-${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}" CACHE PATH "where ${PROJECT_NAME} will be installed" FORCE)
endif()

string(TOUPPER ${PROJECT_NAME} PROJECT_NAME_UPPER)
string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWER)

# 配置构建树的选项
option(${PROJECT_NAME_UPPER}_BUILD_TESTS "Enable testsuite" ON)
option(${PROJECT_NAME_UPPER}_BUILD_DOCS "Build Docs" OFF)
option(${PROJECT_NAME_UPPER}_BUILD_SAMPLES "Build Samples" ON)
option(${PROJECT_NAME_UPPER}_ENABLE_SANITIZER "Enabled sanitized build" OFF)
option(${PROJECT_NAME_UPPER}_BUILD_BENCHMARK "Build Benchmark" ON)
option(${PROJECT_NAME_UPPER}_BUILD_PYTHON "Build Python bindings" ON)

# 配置构建树
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" )

include(ConfigVersion)
include(ConfigBuildTree)
include(ConfigCompiler)
include(ConfigCUDA)
include(ConfigOpenCV)
if(${PROJECT_NAME_UPPER}_BUILD_PYTHON)
    include(ConfigPython)
endif()
include(AddPluginLibrary)
include(AddSample)
include(AddTest)
include(AddBenchmark)
include(ConfigTensorRT)
include(ConfigONNXRuntime)
include(ConfigOpenVINO)
include(ConfigFaiss)

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

set(INFERRT_INSTALL_CMAKEDIR "cmake" CACHE STRING
    "InferRT CMake package install directory" FORCE)

install(EXPORT ${PROJECT_NAME}Targets
    FILE "${PROJECT_NAME}Targets.cmake"
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION "${INFERRT_INSTALL_CMAKEDIR}"
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION "${INFERRT_INSTALL_CMAKEDIR}"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindTensorRT.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ConfigONNXRuntime.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ConfigOpenVINO.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ConfigFaiss.cmake"
    DESTINATION "${INFERRT_INSTALL_CMAKEDIR}"
)

message(STATUS "")
include(PrintConfig)
