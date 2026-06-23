# features 模块依赖 Faiss
if(NOT Faiss_LIBS)
    message(FATAL "Faiss not found, can't build features module")
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
        inferrt_model
        inferrt_ops
)
