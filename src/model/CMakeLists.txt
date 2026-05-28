set(INFERRT_MODEL_PRIVATE_LIBS)
set(INFERRT_MODEL_PUBLIC_LIBS
    opencv_core
    opencv_imgproc
    inferrt_core
    CUDA::cudart
    TensorRT::TensorRT
)

if(INFERRT_WITH_ONNXRUNTIME)
    list(APPEND INFERRT_MODEL_PRIVATE_LIBS ONNXRuntime::ONNXRuntime)
endif()

if(INFERRT_WITH_OPENVINO)
    list(APPEND INFERRT_MODEL_PRIVATE_LIBS openvino::runtime)
endif()

add_plugin_library(model
    PRIVATE_LIBS
        ${INFERRT_MODEL_PRIVATE_LIBS}
    PUBLIC_LIBS
        ${INFERRT_MODEL_PUBLIC_LIBS}
)

target_compile_definitions(inferrt_model
    PRIVATE
        INFERRT_WITH_ONNXRUNTIME=$<BOOL:${INFERRT_WITH_ONNXRUNTIME}>
        INFERRT_WITH_OPENVINO=$<BOOL:${INFERRT_WITH_OPENVINO}>
)

if(INFERRT_WITH_ONNXRUNTIME AND ONNXRUNTIME_DLL)
    target_compile_definitions(inferrt_model PRIVATE INFERRT_ONNXRUNTIME_DLL_PATH="${ONNXRUNTIME_DLL}")
endif()

if(INFERRT_WITH_ONNXRUNTIME AND MSVC)
    target_link_options(inferrt_model PRIVATE /DELAYLOAD:onnxruntime.dll)
    target_link_libraries(inferrt_model PRIVATE delayimp)
endif()

if(INFERRT_WITH_OPENVINO)
    foreach(_ov_target
            openvino_onnx_frontend
            openvino_ir_frontend
            openvino_intel_cpu_plugin
            openvino_intel_gpu_plugin)
        if(TARGET ${_ov_target})
            add_dependencies(inferrt_model ${_ov_target})
        endif()
    endforeach()
endif()

if(INFERRT_WITH_ONNXRUNTIME AND WIN32)
    add_custom_command(TARGET inferrt_model POST_BUILD
        COMMAND ${CMAKE_COMMAND}
                -DINFERRT_COPY_SRC_DIR="${ONNXRUNTIME_BIN_DIR}"
                -DINFERRT_COPY_DST_DIR="$<TARGET_FILE_DIR:inferrt_model>"
                -P "${PROJECT_SOURCE_DIR}/cmake/CopyRuntimeFiles.cmake")
endif()

if(INFERRT_WITH_OPENVINO AND WIN32 AND INFERRT_OPENVINO_BIN_ROOT)
    add_custom_command(TARGET inferrt_model POST_BUILD
        COMMAND ${CMAKE_COMMAND}
                -DINFERRT_COPY_SRC_DIR="${INFERRT_OPENVINO_BIN_ROOT}/$<CONFIG>"
                -DINFERRT_COPY_DST_DIR="$<TARGET_FILE_DIR:inferrt_model>"
                -P "${PROJECT_SOURCE_DIR}/cmake/CopyRuntimeFiles.cmake")
endif()

if(INFERRT_WITH_OPENVINO AND WIN32 AND INFERRT_OPENVINO_TBB_BIN_DIR)
    add_custom_command(TARGET inferrt_model POST_BUILD
        COMMAND ${CMAKE_COMMAND}
                -DINFERRT_COPY_SRC_DIR="${INFERRT_OPENVINO_TBB_BIN_DIR}"
                -DINFERRT_COPY_DST_DIR="$<TARGET_FILE_DIR:inferrt_model>"
                -P "${PROJECT_SOURCE_DIR}/cmake/CopyRuntimeFiles.cmake")
endif()
