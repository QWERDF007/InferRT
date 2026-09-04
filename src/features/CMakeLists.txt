# features 模块依赖 Faiss
if(NOT Faiss_LIBS)
    message(FATAL "Faiss not found, can't build features module")
endif()

# v0 保持可在非 SIMD CPU 上运行；v1/v2 的指令集只作用于各自的私有实现文件，避免
# AVX2/AVX512 要求扩散到公共流程、示例和其他模块。
if(MSVC)
    set_source_files_properties(priv/ShapeTemplateMatcherFastImpl.cpp
        PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(priv/ShapeTemplateMatcherAvx512Impl.cpp
        PROPERTIES
            COMPILE_OPTIONS "/arch:AVX512"
            COMPILE_DEFINITIONS "INFERRT_SHAPE_TEMPLATE_AVX512_ENABLED=1")
else()
    set_source_files_properties(priv/ShapeTemplateMatcherFastImpl.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx2")
    set_source_files_properties(priv/ShapeTemplateMatcherAvx512Impl.cpp
        PROPERTIES
            COMPILE_OPTIONS "-mavx512f;-mavx512bw"
            COMPILE_DEFINITIONS "INFERRT_SHAPE_TEMPLATE_AVX512_ENABLED=1")
endif()

# 图像检索需要 OpenCV 做图片读取/预处理，CUDA 支持 TensorRT/GPU Faiss 路径，
# inferrt_model 提供特征提取模型抽象。
add_plugin_library(features
    PRIVATE_INCS
        ${Faiss_INCLUDE_DIRS}

    PRIVATE_LIBS
        yaml-cpp::yaml-cpp
        inferrt_cvcuda

    PUBLIC_LIBS
        opencv_core opencv_imgcodecs opencv_imgproc
        CUDA::cudart
        Faiss::faiss

        inferrt_core
        inferrt_util
        inferrt_model
        inferrt_ops
)
