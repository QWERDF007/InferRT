# ONNX Runtime 发现与导入：
# - 定位 C++ API 头文件和链接库；
# - 创建 ONNXRuntime::ONNXRuntime 导入目标；
if(TARGET ONNXRuntime::ONNXRuntime)
    return()
endif()

if(NOT DEFINED ONNXRUNTIME_ROOT)
    if(DEFINED ENV{ONNXRUNTIME_ROOT})
        set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "ONNX Runtime installation directory")
    elseif(DEFINED ONNXRuntime_ROOT)
        set(ONNXRUNTIME_ROOT "${ONNXRuntime_ROOT}" CACHE PATH "ONNX Runtime installation directory")
    elseif(DEFINED ENV{ONNXRuntime_ROOT})
        set(ONNXRUNTIME_ROOT "$ENV{ONNXRuntime_ROOT}" CACHE PATH "ONNX Runtime installation directory")
    endif()
endif()

set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIBRARY_DIR "${ONNXRUNTIME_ROOT}/lib")

# 在指定根目录下查找 C++ API 头文件。
find_path(ONNXRUNTIME_INCLUDE_DIR_FOUND
    NAMES onnxruntime_cxx_api.h
    HINTS "${ONNXRUNTIME_INCLUDE_DIR}"
    PATH_SUFFIXES include
)

# 在指定根目录下查找链接库；Windows 下为 .lib，Linux 下为 .so。
find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS "${ONNXRUNTIME_LIBRARY_DIR}"
    PATH_SUFFIXES lib lib64
)

# ONNX Runtime 检查
if(NOT ONNXRUNTIME_INCLUDE_DIR_FOUND OR NOT ONNXRUNTIME_LIBRARY)
    if(INFERRT_BUILD_ONNX)
        message(FATAL_ERROR
            "ONNX Runtime is required when INFERRT_BUILD_ONNX=ON. "
            "Set ONNXRUNTIME_ROOT to a directory containing include/ and lib/. "
            "Current ONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}")
    else()
        return()
    endif()
endif()

# 创建导入目标。
if(WIN32)
    add_library(ONNXRuntime::ONNXRuntime SHARED IMPORTED GLOBAL)
else()
    add_library(ONNXRuntime::ONNXRuntime UNKNOWN IMPORTED GLOBAL)
endif()

if(WIN32)
    get_filename_component(_onnxruntime_library_dir "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
    find_file(ONNXRUNTIME_RUNTIME_LIBRARY
        NAMES onnxruntime.dll
        HINTS "${_onnxruntime_library_dir}" "${ONNXRUNTIME_ROOT}/lib"
        NO_DEFAULT_PATH)
    if(NOT ONNXRUNTIME_RUNTIME_LIBRARY)
        message(FATAL_ERROR
            "ONNX Runtime import library was found, but its matching onnxruntime.dll was not found. "
            "Expected it beside ${ONNXRUNTIME_LIBRARY}")
    endif()
endif()

if(WIN32)
    set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
        IMPORTED_IMPLIB "${ONNXRUNTIME_LIBRARY}"
        IMPORTED_LOCATION "${ONNXRUNTIME_RUNTIME_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR_FOUND}")
else()
    set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR_FOUND}")
endif()

unset(_onnxruntime_library_dir)
