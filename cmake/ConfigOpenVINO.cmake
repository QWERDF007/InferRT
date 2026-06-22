# OpenVINO Runtime 发现与导入：
# - 定位 OpenVINO CMake package；
# - 确保 openvino::runtime 目标可用；
# - tools/*.py 也会读取 INFERRT_OPENVINO_ROOT，用同一份配置收集运行时 DLL。
set(INFERRT_OPENVINO_PROVIDER "required")
set(INFERRT_OPENVINO_ROOT "/home/pc/workspace/openvino_2026.2.0" CACHE PATH "OpenVINO toolkit root directory" FORCE)

# 清除历史缓存，避免修改 INFERRT_OPENVINO_ROOT 后仍沿用旧路径。
unset(INFERRT_OPENVINO_BIN_ROOT CACHE)
unset(INFERRT_OPENVINO_TBB_BIN_DIR CACHE)

# 未显式设置 OpenVINO_DIR 时，从 toolkit 根目录推导 CMake package 路径。
if(NOT OpenVINO_DIR AND EXISTS "${INFERRT_OPENVINO_ROOT}/runtime/cmake/OpenVINOConfig.cmake")
    set(OpenVINO_DIR "${INFERRT_OPENVINO_ROOT}/runtime/cmake" CACHE PATH
        "OpenVINO CMake package directory" FORCE)
endif()

# 优先复用已有目标，否则通过 find_package 加载。
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

# OpenVINO 是强制依赖，未找到时中止配置。
if(NOT TARGET openvino::runtime)
    message(FATAL_ERROR
        "OpenVINO Runtime is required. "
        "Set INFERRT_OPENVINO_ROOT to a toolkit root containing runtime/cmake, "
        "or set OpenVINO_DIR to an installed OpenVINO Runtime CMake package. "
        "Current INFERRT_OPENVINO_ROOT=${INFERRT_OPENVINO_ROOT}; OpenVINO_DIR=${OpenVINO_DIR}")
endif()

# OpenVINO binary packages ship TBB next to the runtime. The OpenVINO CMake package
# records it as an imported dependent library, but consumers still need it on the
# executable link line so ld can resolve libopenvino.so's TBB symbols.
set(_INFERRT_OPENVINO_TBB_DIR "${INFERRT_OPENVINO_ROOT}/runtime/3rdparty/tbb/lib/cmake/TBB")
if(NOT TARGET TBB::tbb AND EXISTS "${_INFERRT_OPENVINO_TBB_DIR}/TBBConfig.cmake")
    find_package(TBB CONFIG QUIET PATHS "${_INFERRT_OPENVINO_TBB_DIR}" NO_DEFAULT_PATH)
endif()

if(TARGET TBB::tbb)
    target_link_libraries(openvino::runtime INTERFACE TBB::tbb)
endif()
unset(_INFERRT_OPENVINO_TBB_DIR)
