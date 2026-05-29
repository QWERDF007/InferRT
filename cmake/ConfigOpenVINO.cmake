# OpenVINO 发现与导入：定位 CMake 包，并确保 openvino::runtime 目标可用。

# 记录 OpenVINO 来源：required（待探测）/ preloaded（已预加载）/ package（find_package 找到）。
set(INFERRT_OPENVINO_PROVIDER "required")
# 用户可通过 -DINFERRT_OPENVINO_ROOT=... 覆盖默认 toolkit 根目录。
set(INFERRT_OPENVINO_ROOT "D:/Software/openvino_toolkit" CACHE PATH "OpenVINO toolkit root directory")

# 清除历史缓存，避免修改 INFERRT_OPENVINO_ROOT 后仍沿用旧路径。
unset(INFERRT_OPENVINO_BIN_ROOT CACHE)
unset(INFERRT_OPENVINO_TBB_BIN_DIR CACHE)

# 若未显式设置 OpenVINO_DIR，则从 toolkit 根目录推断 CMake 包路径。
if(NOT OpenVINO_DIR AND EXISTS "${INFERRT_OPENVINO_ROOT}/runtime/cmake/OpenVINOConfig.cmake")
    set(OpenVINO_DIR "${INFERRT_OPENVINO_ROOT}/runtime/cmake" CACHE PATH
        "OpenVINO CMake package directory" FORCE)
endif()

# 查找 openvino::runtime 目标：优先使用已存在的导入目标，否则通过 find_package 加载。
if(TARGET openvino::runtime)
    set(INFERRT_OPENVINO_PROVIDER "preloaded")
else()
    find_package(OpenVINO QUIET COMPONENTS Runtime ONNX)
    if(NOT OpenVINO_FOUND)
        find_package(openvino QUIET)
    endif()

    if(TARGET openvino::runtime)
        set(INFERRT_OPENVINO_PROVIDER "package")
    endif()
endif()

# OpenVINO 为必需依赖，未找到时中止配置。
if(NOT TARGET openvino::runtime)
    message(FATAL_ERROR
        "OpenVINO Runtime is required. "
        "Set INFERRT_OPENVINO_ROOT to a toolkit root containing runtime/cmake, "
        "or set OpenVINO_DIR to an installed OpenVINO Runtime CMake package. "
        "Current INFERRT_OPENVINO_ROOT=${INFERRT_OPENVINO_ROOT}; OpenVINO_DIR=${OpenVINO_DIR}")
endif()
