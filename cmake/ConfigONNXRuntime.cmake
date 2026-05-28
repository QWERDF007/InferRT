set(ONNXRUNTIME_ROOT "D:/Software/onnxruntime-gpu-1.20.1" CACHE PATH "ONNX Runtime installation directory")

set(INFERRT_WITH_ONNXRUNTIME OFF)

if(INFERRT_ENABLE_ONNXRUNTIME)
    set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
    set(ONNXRUNTIME_LIBRARY_DIR "${ONNXRUNTIME_ROOT}/lib")
    unset(ONNXRUNTIME_INCLUDE_DIR_FOUND CACHE)
    unset(ONNXRUNTIME_LIBRARY CACHE)
    unset(ONNXRUNTIME_DLL CACHE)

    find_path(ONNXRUNTIME_INCLUDE_DIR_FOUND
        NAMES onnxruntime_cxx_api.h
        HINTS "${ONNXRUNTIME_INCLUDE_DIR}"
        NO_DEFAULT_PATH)

    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        HINTS "${ONNXRUNTIME_LIBRARY_DIR}"
        NO_DEFAULT_PATH)
    find_file(ONNXRUNTIME_DLL
        NAMES onnxruntime.dll
        HINTS "${ONNXRUNTIME_LIBRARY_DIR}"
        NO_DEFAULT_PATH)

    if(NOT ONNXRUNTIME_INCLUDE_DIR_FOUND OR NOT ONNXRUNTIME_LIBRARY)
        message(FATAL_ERROR
            "ONNX Runtime was requested but not found. "
            "Set ONNXRUNTIME_ROOT to a directory containing include/ and lib/. "
            "Current ONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}")
    endif()

    add_library(ONNXRuntime::ONNXRuntime SHARED IMPORTED)
    set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
        IMPORTED_IMPLIB "${ONNXRUNTIME_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR_FOUND}")
    if(ONNXRUNTIME_DLL)
        set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES IMPORTED_LOCATION "${ONNXRUNTIME_DLL}")
    else()
        set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}")
    endif()

    set(ONNXRUNTIME_BIN_DIR "${ONNXRUNTIME_LIBRARY_DIR}" CACHE PATH "ONNX Runtime DLL directory" FORCE)
    set(INFERRT_WITH_ONNXRUNTIME ON)
endif()
