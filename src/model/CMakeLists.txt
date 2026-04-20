add_plugin_library(model
    PRIVATE_LIBS
        # 在这里添加 PRIVATE 链接的库
    PUBLIC_LIBS
        # 在这里添加 PUBLIC 链接的库
        opencv_core
        inferrt_core
        CUDA::cudart
        TensorRT::TensorRT
)