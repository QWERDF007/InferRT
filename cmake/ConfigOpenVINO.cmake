set(INFERRT_WITH_OPENVINO OFF)
set(INFERRT_OPENVINO_PROVIDER "disabled")
set(INFERRT_OPENVINO_ROOT "D:/Software/openvino_toolkit" CACHE PATH "OpenVINO toolkit root directory")

if(NOT OpenVINO_DIR AND EXISTS "${INFERRT_OPENVINO_ROOT}/runtime/cmake/OpenVINOConfig.cmake")
    set(OpenVINO_DIR "${INFERRT_OPENVINO_ROOT}/runtime/cmake" CACHE PATH
        "OpenVINO CMake package directory" FORCE)
endif()

if(EXISTS "${INFERRT_OPENVINO_ROOT}/runtime/bin/intel64")
    set(INFERRT_OPENVINO_BIN_ROOT "${INFERRT_OPENVINO_ROOT}/runtime/bin/intel64" CACHE PATH
        "OpenVINO runtime DLL root directory" FORCE)
endif()

if(EXISTS "${INFERRT_OPENVINO_ROOT}/runtime/3rdparty/tbb/bin")
    set(INFERRT_OPENVINO_TBB_BIN_DIR "${INFERRT_OPENVINO_ROOT}/runtime/3rdparty/tbb/bin" CACHE PATH
        "OpenVINO TBB runtime DLL directory" FORCE)
endif()

if(INFERRT_ENABLE_OPENVINO)
    if(TARGET openvino::runtime)
        set(INFERRT_WITH_OPENVINO ON)
        set(INFERRT_OPENVINO_PROVIDER "preloaded")
    else()
        find_package(OpenVINO QUIET COMPONENTS Runtime ONNX)
        if(NOT OpenVINO_FOUND)
            find_package(openvino QUIET)
        endif()

        if(TARGET openvino::runtime)
            set(INFERRT_WITH_OPENVINO ON)
            set(INFERRT_OPENVINO_PROVIDER "package")
        endif()
    endif()

    if(NOT INFERRT_WITH_OPENVINO)
        message(FATAL_ERROR
            "OpenVINO backend was requested but OpenVINO Runtime was not found. "
            "Set INFERRT_OPENVINO_ROOT to a toolkit root containing runtime/cmake, "
            "or set OpenVINO_DIR to an installed OpenVINO Runtime CMake package. "
            "Current INFERRT_OPENVINO_ROOT=${INFERRT_OPENVINO_ROOT}; OpenVINO_DIR=${OpenVINO_DIR}")
    endif()
endif()
