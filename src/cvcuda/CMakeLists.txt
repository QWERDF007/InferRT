add_plugin_library(cvcuda
    PRIVATE_LIBS
        # 在这里添加 PRIVATE 链接的库
    PUBLIC_LIBS
        # 在这里添加 PUBLIC 链接的库
        cudart
        opencv_core
        inferrt_core
        inferrt_util
)
