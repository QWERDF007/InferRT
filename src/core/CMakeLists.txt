add_plugin_library(core
    PRIVATE_LIBS
        # 在这里添加 PRIVATE 链接的库
    PUBLIC_LIBS
        # 在这里添加 PUBLIC 链接的库
)

configure_version(inferrt_core INFERRT ${PROJECT_NAME_LOWER}/core ${PROJECT_VERSION})
