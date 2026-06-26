add_plugin_library(model
    PRIVATE_LIBS

    PUBLIC_LIBS
        opencv_core opencv_imgproc
        CUDA::cudart
        TensorRT::TensorRT
        ONNXRuntime::ONNXRuntime
        openvino::runtime

        inferrt_core
        inferrt_util
)

# 确保 OpenVINO 前端与 CPU/GPU 插件在 inferrt_model 之前完成构建，避免链接时缺少对应产物。
foreach(_ov_target
        openvino_onnx_frontend
        openvino_ir_frontend
        openvino_intel_cpu_plugin
        openvino_intel_gpu_plugin)
    if(TARGET ${_ov_target})
        add_dependencies(inferrt_model ${_ov_target})
    endif()
endforeach()
