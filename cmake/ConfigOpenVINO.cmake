# OpenVINO Runtime 发现与导入：
# - 定位 OpenVINO CMake package；
# - 确保 openvino::runtime 目标可用；
set(INFERRT_OPENVINO_PROVIDER "required")

if(NOT DEFINED INFERRT_OPENVINO_ROOT)
    if(DEFINED ENV{INFERRT_OPENVINO_ROOT})
        set(INFERRT_OPENVINO_ROOT "$ENV{INFERRT_OPENVINO_ROOT}" CACHE PATH "OpenVINO toolkit root directory")
    elseif(DEFINED OpenVINO_ROOT)
        set(INFERRT_OPENVINO_ROOT "${OpenVINO_ROOT}" CACHE PATH "OpenVINO toolkit root directory")
    elseif(DEFINED ENV{OpenVINO_ROOT})
        set(INFERRT_OPENVINO_ROOT "$ENV{OpenVINO_ROOT}" CACHE PATH "OpenVINO toolkit root directory")
    endif()
endif()

# 未显式设置 OpenVINO_DIR 时，从 toolkit 根目录推导 CMake package 路径。
if(NOT OpenVINO_DIR AND DEFINED INFERRT_OPENVINO_ROOT AND EXISTS "${INFERRT_OPENVINO_ROOT}/runtime/cmake/OpenVINOConfig.cmake")
    set(OpenVINO_DIR "${INFERRT_OPENVINO_ROOT}/runtime/cmake" CACHE PATH "OpenVINO CMake package directory")
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

# OpenVINO 检查
if(NOT TARGET openvino::runtime)
    if(INFERRT_BUILD_OPENVINO)
        message(FATAL_ERROR
            "OpenVINO Runtime is required when INFERRT_BUILD_OPENVINO=ON. "
            "Set INFERRT_OPENVINO_ROOT to a toolkit root containing runtime/cmake, "
            "or set OpenVINO_DIR to an installed OpenVINO Runtime CMake package. "
            "Current INFERRT_OPENVINO_ROOT=${INFERRT_OPENVINO_ROOT}; OpenVINO_DIR=${OpenVINO_DIR}")
    else()
        return()
    endif()
endif()

if(DEFINED INFERRT_OPENVINO_ROOT)
    set(_INFERRT_OPENVINO_TBB_DIR "${INFERRT_OPENVINO_ROOT}/runtime/3rdparty/tbb/lib/cmake/TBB")
    if(NOT TARGET TBB::tbb AND EXISTS "${_INFERRT_OPENVINO_TBB_DIR}/TBBConfig.cmake")
        find_package(TBB CONFIG QUIET PATHS "${_INFERRT_OPENVINO_TBB_DIR}" NO_DEFAULT_PATH)
    endif()

    if(TARGET TBB::tbb)
        target_link_libraries(openvino::runtime INTERFACE TBB::tbb)
    endif()
    unset(_INFERRT_OPENVINO_TBB_DIR)
endif()
