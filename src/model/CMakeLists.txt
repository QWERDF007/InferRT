set(_INFERRT_MODEL_PUBLIC_LIBS
    opencv_core opencv_imgproc
    CUDA::cudart
    inferrt_core
    inferrt_util
)
set(_INFERRT_MODEL_PRIVATE_LIBS
    TensorRT::TensorRT
)
set(_INFERRT_MODEL_EXCLUDE_SOURCES)

if(${PROJECT_NAME_UPPER}_BUILD_ONNX)
    list(APPEND _INFERRT_MODEL_PRIVATE_LIBS ONNXRuntime::ONNXRuntime)
else()
    list(APPEND _INFERRT_MODEL_EXCLUDE_SOURCES priv/ONNXRuntimeBackend.cpp)
endif()

if(${PROJECT_NAME_UPPER}_BUILD_OPENVINO)
    list(APPEND _INFERRT_MODEL_PRIVATE_LIBS openvino::runtime)
else()
    list(APPEND _INFERRT_MODEL_EXCLUDE_SOURCES priv/OpenVINOBackend.cpp)
endif()

add_plugin_library(model
    EXCLUDE_SOURCES ${_INFERRT_MODEL_EXCLUDE_SOURCES}
    PUBLIC_LIBS ${_INFERRT_MODEL_PUBLIC_LIBS}
    PRIVATE_LIBS ${_INFERRT_MODEL_PRIVATE_LIBS}
)

# 确保 OpenVINO 前端与 CPU/GPU 插件在 inferrt_model 之前完成构建，避免链接时缺少对应产物。
if(${PROJECT_NAME_UPPER}_BUILD_OPENVINO)
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
