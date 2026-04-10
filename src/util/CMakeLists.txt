add_plugin_library(util
    PRIVATE_LIBS
        # 在这里添加 PRIVATE 链接的库
    PUBLIC_LIBS
        # 在这里添加 PUBLIC 链接的库
        cudart
        inferrt_core
)

# 只在 DEBUG 构建时暴露源代码信息（文件名、行号、调用语句）
target_compile_definitions(inferrt_util PUBLIC $<$<CONFIG:Debug>:IRT_EXPOSE_CODE=1>)