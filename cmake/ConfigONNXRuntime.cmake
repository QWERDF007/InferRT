# ONNX Runtime 发现与导入：定位头文件/库，并创建 ONNXRuntime::ONNXRuntime 导入目标。

# 用户可通过 -DONNXRUNTIME_ROOT=... 覆盖默认安装路径。
set(ONNXRUNTIME_ROOT "D:/Software/onnxruntime-gpu-1.19.0" CACHE PATH "ONNX Runtime installation directory")

# 清除历史缓存，避免修改 ONNXRUNTIME_ROOT 后仍沿用旧的探测结果。
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

# 在指定根目录下查找链接库（Windows 为 .lib，Linux 为 .so）。
find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS "${ONNXRUNTIME_LIBRARY_DIR}"
    NO_DEFAULT_PATH)

if(NOT ONNXRUNTIME_INCLUDE_DIR_FOUND OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR
        "ONNX Runtime is required. "
        "Set ONNXRUNTIME_ROOT to a directory containing include/ and lib/. "
        "Current ONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}")
endif()

# 创建 CMake 导入目标，供 inferrt_model 等模块链接；运行时 DLL 由后端代码按需加载。
add_library(ONNXRuntime::ONNXRuntime UNKNOWN IMPORTED)
set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR_FOUND}")
