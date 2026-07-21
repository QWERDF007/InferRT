# features 模块依赖 Faiss
if(NOT Faiss_LIBS)
    message(FATAL "Faiss not found, can't build features module")
endif()

# v0 保持可在非 AVX2 CPU 上运行；只为 v1 快速实现启用 AVX2，避免把该 ISA
# 要求扩散到公共流程、示例和其他模块。未来 AVX512 版本可按同样方式增加独立快速实现。
if(MSVC)
    set_source_files_properties(priv/ShapeTemplateMatcherFastImpl.cpp
        PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
    set_source_files_properties(priv/ShapeTemplateMatcherFastImpl.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx2")
endif()

# 图像检索需要 OpenCV 做图片读取/预处理，CUDA 支持 TensorRT/GPU Faiss 路径，
# inferrt_model 提供特征提取模型抽象。
add_plugin_library(features
    PRIVATE_INCS
        ${Faiss_INCLUDE_DIRS}

    PRIVATE_LIBS

    PUBLIC_LIBS
        opencv_core opencv_imgcodecs opencv_imgproc
        CUDA::cudart
        ${Faiss_LIBS}

        inferrt_core
        inferrt_util
        inferrt_model
        inferrt_ops
)
