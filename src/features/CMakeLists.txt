# features 模块依赖 Faiss；未找到 Faiss 时跳过整个图像检索库和相关样例/测试。
if(NOT Faiss_LIBS)
    message(STATUS "Faiss not found, skipping features module")
    return()
endif()

# 图像检索需要 OpenCV 做图片读取/预处理，CUDA 支持 TensorRT/GPU Faiss 路径，
# inferrt_model 提供特征提取模型抽象。
add_plugin_library(features
    PRIVATE_LIBS

    PUBLIC_LIBS
        opencv_core opencv_imgcodecs opencv_imgproc
        CUDA::cudart
        ${Faiss_LIBS}

        inferrt_core
        inferrt_model
        inferrt_ops
)

target_include_directories(${PROJECT_NAME_LOWER}_features
    PRIVATE
        ${Faiss_INCLUDE_DIRS}
)
