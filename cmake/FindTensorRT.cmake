# FindTensorRT.cmake - 用于查找和配置 TensorRT 库的 CMake 模块
# 此模块会自动检测 TensorRT 的安装路径，并配置相应的库和头文件

cmake_minimum_required(VERSION 3.17.0)

if(TARGET TensorRT)
  return()
endif()

# 内部辅助函数：根据必需文件猜测有效路径
# 参数：
#   var_name - 用于存储结果的变量名
#   required_files - 必须存在的文件列表
#   ARGN - 候选路径列表
function(_guess_path var_name required_files)
   set(_result "")

  # 遍历所有候选路径
  foreach(path_entry IN LISTS ARGN)
    # 跳过不存在的路径
    if(NOT EXISTS "${path_entry}")
      message(DEBUG "skip non-existing path '${path_entry}'")
      continue()
    endif()

    # 检查该路径是否包含所有必需文件
    set(_ok TRUE)
    foreach(required_file IN LISTS required_files)
      if(NOT EXISTS "${path_entry}/${required_file}")
        set(_ok FALSE)
        message(DEBUG "'${path_entry}' missing '${required_file}'")
        break()
      endif()
    endforeach()

    # 如果路径有效，添加到结果列表
    if(_ok)
      list(APPEND _result "${path_entry}")
      message(DEBUG "accept '${path_entry}'")
    else()
      message(DEBUG "reject '${path_entry}'")
    endif()
  endforeach()

  # 如果没有找到有效路径，报错
  if(_result STREQUAL "")
    message(
      FATAL_ERROR
        "_guess_path(${var_name}) failed: no valid path found. required_files='${required_files}' candidates='${ARGN}'"
    )
  endif()

  # 将结果返回给父作用域
  set(${var_name}
      "${_result}"
      PARENT_SCOPE)
endfunction()

# 创建 TensorRT 导入接口库
# IMPORTED INTERFACE 表示这是一个外部库的接口目标
add_library(TensorRT INTERFACE IMPORTED GLOBAL)
# 创建别名，允许使用 TensorRT::TensorRT 命名空间形式
add_library(TensorRT::TensorRT ALIAS TensorRT)


# Resolve an explicit root first, then fall back to standard CMake prefix
# search paths.  A clean build must not depend on a stale cache entry.
if(NOT DEFINED TRT_ROOT OR TRT_ROOT STREQUAL "")
  set(_trt_root_hints)
  if(DEFINED TensorRT_ROOT AND NOT TensorRT_ROOT STREQUAL "")
    list(APPEND _trt_root_hints "${TensorRT_ROOT}")
  endif()
  foreach(_trt_root_environment TRT_ROOT TensorRT_ROOT TENSORRT_ROOT)
    if(DEFINED ENV{${_trt_root_environment}} AND NOT "$ENV{${_trt_root_environment}}" STREQUAL "")
      list(APPEND _trt_root_hints "$ENV{${_trt_root_environment}}")
    endif()
  endforeach()
  if(CMAKE_PREFIX_PATH)
    list(APPEND _trt_root_hints ${CMAKE_PREFIX_PATH})
  endif()
  if(DEFINED INFERRT_TENSORRT_DEFAULT_ROOT
     AND NOT INFERRT_TENSORRT_DEFAULT_ROOT STREQUAL "")
    list(APPEND _trt_root_hints "${INFERRT_TENSORRT_DEFAULT_ROOT}")
  endif()

  find_path(_trt_include_dir
    NAMES NvInfer.h
    HINTS ${_trt_root_hints}
    PATH_SUFFIXES include)
  if(_trt_include_dir)
    get_filename_component(TRT_ROOT "${_trt_include_dir}" DIRECTORY)
  endif()
  unset(_trt_include_dir CACHE)
endif()

if(NOT DEFINED TRT_ROOT OR TRT_ROOT STREQUAL "")
  message(
    FATAL_ERROR
      "TensorRT was not found. Set -DTRT_ROOT=/path/to/tensorrt, "
      "TensorRT_ROOT, or add the TensorRT prefix to CMAKE_PREFIX_PATH."
  )
endif()

if(NOT EXISTS "${TRT_ROOT}")
  message(
    FATAL_ERROR
      "TRT_ROOT=${TRT_ROOT} does not exist. Check the TensorRT installation path."
  )
endif()

message(STATUS "Using TRT_ROOT: ${TRT_ROOT}")

# TensorRT 版本号配置
# 支持的格式示例：
#   - "8.6.1.6"
#   - "8.6.1.6+cuda12.0.1.011"
#   - "8.6.1.6.Windows10.x86_64.cuda-12.0"
set(TRT_VERSION
    ""
    CACHE
      STRING
      "TensorRT version, e.g. \"8.6.1.6\" or \"8.6.1.6+cuda12.0.1.011\", \"8.6.1.6.Windows10.x86_64.cuda-12.0\" etc"
)

# 如果 CMake 变量和环境变量都定义了，优先使用环境变量
if(NOT TRT_VERSION STREQUAL "" AND NOT "$ENV{TRT_VERSION}" STREQUAL "")
  message(
    WARNING
      "TRT_VERSION defined by cmake and environment variable both, using the later one"
  )
endif()

if(NOT "$ENV{TRT_VERSION}" STREQUAL "")
  set(TRT_VERSION "$ENV{TRT_VERSION}")
endif()

# 安装包的消费者通常只提供 TRT_ROOT，不会有构建树里的 TRT_VERSION 缓存。
# 优先从 TensorRT 版本头文件推导完整版本；Windows 版头文件可能通过
# TRT_*_ENTERPRISE 宏间接定义 NV_TENSORRT_*，因此同时支持两种写法。
file(TO_CMAKE_PATH "${TRT_ROOT}" _trt_root_cmake)
if(TRT_VERSION STREQUAL "")
  file(GLOB_RECURSE _trt_version_headers CONFIGURE_DEPENDS LIST_DIRECTORIES FALSE
       "${_trt_root_cmake}/*NvInferVersion.h")
  list(SORT _trt_version_headers)
  foreach(_trt_version_header IN LISTS _trt_version_headers)
    if(EXISTS "${_trt_version_header}")
      unset(_trt_version_MAJOR)
      unset(_trt_version_MINOR)
      unset(_trt_version_PATCH)
      unset(_trt_version_BUILD)
      file(READ "${_trt_version_header}" _trt_version_text)
      foreach(_trt_version_component MAJOR MINOR PATCH BUILD)
        string(REGEX MATCH
               "#[ \t]*define[ \t]+NV_TENSORRT_${_trt_version_component}[ \t]+([0-9]+)"
               _trt_component_match "${_trt_version_text}")
        if(NOT _trt_component_match)
          string(REGEX MATCH
                 "#[ \t]*define[ \t]+TRT_${_trt_version_component}_[A-Za-z0-9_]+[ \t]+([0-9]+)"
                 _trt_component_match "${_trt_version_text}")
        endif()
        if(_trt_component_match)
          set(_trt_component_value "${CMAKE_MATCH_1}")
          set(_trt_version_${_trt_version_component} "${_trt_component_value}")
        endif()
      endforeach()
      if(DEFINED _trt_version_MAJOR AND DEFINED _trt_version_MINOR
         AND DEFINED _trt_version_PATCH)
        set(TRT_VERSION "${_trt_version_MAJOR}.${_trt_version_MINOR}.${_trt_version_PATCH}")
        if(DEFINED _trt_version_BUILD)
          string(APPEND TRT_VERSION ".${_trt_version_BUILD}")
        endif()
        break()
      endif()
    endif()
  endforeach()
endif()

# TensorRT 10 的 Windows DLL 名称也包含主版本号，可作为缺少头文件时的
# 明确回退；没有任何可验证版本时给出可操作错误，而不是让 `if()` 解析空值。
if(TRT_VERSION STREQUAL "" AND WIN32)
  file(GLOB _trt_version_libraries CONFIGURE_DEPENDS LIST_DIRECTORIES FALSE
       "${_trt_root_cmake}/bin/nvinfer_*.dll")
  foreach(_trt_version_library IN LISTS _trt_version_libraries)
    get_filename_component(_trt_version_library_name "${_trt_version_library}" NAME)
    string(REGEX MATCH "^nvinfer_([0-9]+)\.dll$" _trt_library_match
           "${_trt_version_library_name}")
    if(_trt_library_match)
      set(TRT_VERSION "${CMAKE_MATCH_1}")
      break()
    endif()
  endforeach()
endif()

if(TRT_VERSION STREQUAL "")
  message(FATAL_ERROR
          "TRT_VERSION is not set and could not be inferred from TRT_ROOT=${TRT_ROOT}. "
          "Set -DTRT_VERSION=<major.minor.patch[.build]> or provide "
          "${TRT_ROOT}/include/NvInferVersion.h.")
endif()

# 从 TRT_VERSION 中提取主版本号（第一个连续的数字序列）
# 例如："8.6.1.6" → "8", "10.0.1.6" → "10"
string(REGEX MATCH "^[0-9]+" TRT_MAJOR_VERSION "${TRT_VERSION}")
if(TRT_MAJOR_VERSION STREQUAL "")
  message(FATAL_ERROR "Invalid TRT_VERSION='${TRT_VERSION}': expected a numeric major version")
endif()


# 根据操作系统和 TensorRT 版本配置库模块
if(WIN32)
  # Windows 平台配置
  # 默认开发运行时只链接完整 TensorRT runtime、plugin 与 ONNX parser。
  # dispatch/lean 是部署用的裁剪运行时，二者导出 stub API；与完整 nvinfer
  # 同时链接会使 createInferRuntime 等符号解析到 stub 实现。
  if(${TRT_MAJOR_VERSION} GREATER_EQUAL 10)
    set(_modules nvinfer_10 nvinfer_plugin_10 nvonnxparser_10)
    message(DEBUG "Using ${_modules}")
  else()
    set(_modules nvinfer nvinfer_plugin nvonnxparser)
  endif()

  # Windows 下的库和头文件路径结构比较简单
  set(TensorRT_LIBRARY_DIR "${TRT_ROOT}/lib")
  set(TensorRT_INCLUDE_DIR "${TRT_ROOT}/include")
elseif(UNIX)
  # Linux/Unix 平台配置
  # dispatch/lean 是面向部署的裁剪 runtime，不属于默认开发链接集。
  set(_modules nvinfer nvinfer_plugin nvonnxparser)

  # 尝试常见的子目录结构
  # 将系统架构转换为小写以便匹配
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _trt_arch)
  if(_trt_arch MATCHES "^(aarch64|arm64|arch64)$")
    # ARM64 架构的候选路径
    set(_trt_include_candidates
        "${TRT_ROOT}/include"
        "${TRT_ROOT}/targets/aarch64-linux/include")
    set(_trt_library_candidates
        "${TRT_ROOT}/lib"
        "${TRT_ROOT}/targets/aarch64-linux-gnu/lib"
        "${TRT_ROOT}/targets/aarch64-linux/lib")
  elseif(_trt_arch MATCHES "^(x86_64|amd64)$")
    # x86_64 架构的候选路径
    set(_trt_include_candidates
        "${TRT_ROOT}/include"
        "${TRT_ROOT}/targets/x86_64-linux-gnu/include"
        "${TRT_ROOT}/targets/x86_64-linux/include")
    set(_trt_library_candidates
        "${TRT_ROOT}/lib"
        "${TRT_ROOT}/targets/x86_64-linux-gnu/lib"
        "${TRT_ROOT}/targets/x86_64-linux/lib")
  else()
    message(FATAL_ERROR "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
  endif()

  # 使用辅助函数查找包含必需库文件的有效路径
  _guess_path(TensorRT_LIBRARY_DIR "libnvinfer.so;libnvinfer_plugin.so"
              ${_trt_library_candidates})
  message(STATUS "TensorRT libraries: ${TensorRT_LIBRARY_DIR}")
  # 查找包含必需头文件的有效路径
  _guess_path(TensorRT_INCLUDE_DIR "NvInfer.h" ${_trt_include_candidates})
  message(STATUS "TensorRT includes: ${TensorRT_INCLUDE_DIR}")
endif()

# 查找所有需要的 TensorRT 库文件
foreach(lib IN LISTS _modules)
  find_library(
    TensorRT_${lib}_LIBRARY
    NAMES ${lib}
    HINTS ${TensorRT_LIBRARY_DIR})
  list(APPEND TensorRT_LIBRARIES ${TensorRT_${lib}_LIBRARY})
endforeach()

# 将找到的库链接到 TensorRT 目标
target_link_libraries(TensorRT INTERFACE ${TensorRT_LIBRARIES})

message(STATUS "Found TensorRT libs: ${TensorRT_LIBRARIES}")

# 设置 TensorRT 目标的属性
set_target_properties(TensorRT PROPERTIES 
  C_STANDARD 17                                            # C 标准版本
  CXX_STANDARD 17                                          # C++ 标准版本
  POSITION_INDEPENDENT_CODE ON                             # 启用位置无关代码（PIC）
  SKIP_BUILD_RPATH TRUE                                    # 构建时跳过 RPATH
  BUILD_WITH_INSTALL_RPATH TRUE                            # 使用安装 RPATH 进行构建
  INSTALL_RPATH "$ORIGIN"                                  # 设置 RPATH 为相对路径（$ORIGIN 表示可执行文件所在目录）
  INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}") # 设置头文件包含目录

# 清理临时变量
unset(TRT_MAJOR_VERSION)
unset(_modules)
unset(_trt_include_candidates)
unset(_trt_library_candidates)
unset(_trt_arch)
unset(_trt_root_cmake)
