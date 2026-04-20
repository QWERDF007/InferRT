# FindTensorRT.cmake - 用于查找和配置 TensorRT 库的 CMake 模块
# 此模块会自动检测 TensorRT 的安装路径，并配置相应的库和头文件

cmake_minimum_required(VERSION 3.17.0)

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
add_library(TensorRT IMPORTED INTERFACE)
# 创建别名，允许使用 TensorRT::TensorRT 命名空间形式
add_library(TensorRT::TensorRT ALIAS TensorRT)


# 检查 TRT_ROOT 是否已设置（CMake 变量或环境变量）
if(NOT DEFINED TRT_ROOT AND NOT DEFINED ENV{TRT_ROOT})
  message(
    FATAL_ERROR
      "TRT_ROOT is not set. Please set TRT_ROOT to the TensorRT installation directory.\n"
      "You can set it via:\n"
      "  - CMake variable: -DTRT_ROOT=/path/to/tensorrt\n"
      "  - Environment variable: set TRT_ROOT=/path/to/tensorrt"
  )
endif()

# 如果 CMake 变量未设置，则使用环境变量
if(NOT DEFINED TRT_ROOT)
  set(TRT_ROOT $ENV{TRT_ROOT})
endif()

# 验证 TRT_ROOT 路径是否存在
if(NOT EXISTS "${TRT_ROOT}")
  message(
    FATAL_ERROR
      "TRT_ROOT=${TRT_ROOT} does not exist! Please check your TensorRT installation path."
  )
endif()

message(STATUS "Using TRT_ROOT: ${TRT_ROOT}")

# TensorRT 版本号配置
# 支持的格式示例：
#   - "8.6.1.6"
#   - "8.6.1.6+cuda12.0.1.011"
#   - "8.6.1.6.Windows10.x86_64.cuda-12.0"
set(TRT_VERSION
    CACHE
      STRING
      "TensorRT version, e.g. \"8.6.1.6\" or \"8.6.1.6+cuda12.0.1.011\", \"8.6.1.6.Windows10.x86_64.cuda-12.0\" etc"
)

# 如果 CMake 变量和环境变量都定义了，优先使用环境变量
if(NOT TRT_VERSION STREQUAL "" AND NOT $ENV{TRT_VERSION} STREQUAL "")
  message(
    WARNING
      "TRT_VERSION defined by cmake and environment variable both, using the later one"
  )
endif()

if(NOT $ENV{TRT_VERSION} STREQUAL "")
  set(TRT_VERSION $ENV{TRT_VERSION})
endif()

# 从 TRT_VERSION 中提取主版本号（第一个连续的数字序列）
# 例如："8.6.1.6" → "8", "10.0.1.6" → "10", "12.5.0.0" → "12"
string(REGEX MATCH "([0-9]+)" _match "${TRT_VERSION}")
set(TRT_MAJOR_VERSION "${_match}")
unset(_match)


# 根据操作系统和 TensorRT 版本配置库模块
if(WIN32)
  # Windows 平台配置
  # TensorRT 10+ 版本的库文件名包含版本号后缀
  if(${TRT_MAJOR_VERSION} GREATER_EQUAL 10)
    set(_modules nvinfer_10 nvinfer_plugin_10 nvinfer_vc_plugin_10
                 nvinfer_dispatch_10 nvinfer_lean_10)
    message(DEBUG "Using ${_modules}")
  else()
    set(_modules nvinfer nvinfer_plugin nvinfer_vc_plugin nvinfer_dispatch
                 nvinfer_lean)
  endif()

  # Windows 下的库和头文件路径结构比较简单
  set(TensorRT_LIBRARY_DIR "${TRT_ROOT}/lib")
  set(TensorRT_INCLUDE_DIR "${TRT_ROOT}/include")
elseif(UNIX)
  # Linux/Unix 平台配置
  set(_modules nvinfer nvinfer_plugin)
  # TensorRT 8+ 版本增加了额外的库模块
  if(${TRT_MAJOR_VERSION} GREATER_EQUAL 8)
    list(APPEND _modules nvinfer_vc_plugin nvinfer_dispatch nvinfer_lean)
  endif()

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
