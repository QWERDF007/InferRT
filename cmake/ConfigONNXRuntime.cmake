# ONNX Runtime 发现与导入：
# - 定位 C++ API 头文件和链接库；
# - 创建 ONNXRuntime::ONNXRuntime 导入目标；
# - tools/*.py 也会读取 ONNXRUNTIME_ROOT，用同一份配置收集运行时 DLL。
if(TARGET ONNXRuntime::ONNXRuntime)
    return()
endif()

set(ONNXRUNTIME_ROOT "/home/pc/workspace/onnxruntime-linux-x64-gpu-1.26.0" CACHE PATH "ONNX Runtime installation directory" FORCE)

# 清除历史缓存，避免修改 ONNXRUNTIME_ROOT 后仍沿用旧探测结果。
unset(ONNXRUNTIME_DLL CACHE)
unset(ONNXRUNTIME_BIN_DIR CACHE)
unset(INFERRT_ONNXRUNTIME_EXTRA_DLL_DIRS CACHE)

set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIBRARY_DIR "${ONNXRUNTIME_ROOT}/lib")
unset(ONNXRUNTIME_INCLUDE_DIR_FOUND CACHE)
unset(ONNXRUNTIME_LIBRARY CACHE)

# 在指定根目录下查找 C++ API 头文件。
find_path(ONNXRUNTIME_INCLUDE_DIR_FOUND
    NAMES onnxruntime_cxx_api.h
    HINTS "${ONNXRUNTIME_INCLUDE_DIR}"
    NO_DEFAULT_PATH)

# 在指定根目录下查找链接库；Windows 下为 .lib，Linux 下为 .so。
find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS "${ONNXRUNTIME_LIBRARY_DIR}"
    NO_DEFAULT_PATH)

# ONNX Runtime 是强制依赖，未找到时中止配置。
if(NOT ONNXRUNTIME_INCLUDE_DIR_FOUND OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR
        "ONNX Runtime is required. "
        "Set ONNXRUNTIME_ROOT to a directory containing include/ and lib/. "
        "Current ONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}")
endif()

# 创建导入目标。运行时 DLL 由系统 loader 解析，不再由 C++ 代码手动加载。
add_library(ONNXRuntime::ONNXRuntime UNKNOWN IMPORTED GLOBAL)
set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR_FOUND}")
