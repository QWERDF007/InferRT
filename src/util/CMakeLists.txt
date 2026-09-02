set(UTIL_PRIVATE_LIBS
    yaml-cpp::yaml-cpp
    cpuinfo
)
set(UTIL_PUBLIC_LIBS
    inferrt_core
)

if(${PROJECT_NAME_UPPER}_ENABLE_CUDA)
    list(APPEND UTIL_PRIVATE_LIBS CUDA::nvml)
    list(APPEND UTIL_PUBLIC_LIBS CUDA::cudart)
endif()

add_plugin_library(util
    PRIVATE_LIBS
        ${UTIL_PRIVATE_LIBS}
    PUBLIC_LIBS
        ${UTIL_PUBLIC_LIBS}
)

if(${PROJECT_NAME_UPPER}_ENABLE_CUDA)
    target_compile_definitions(inferrt_util PUBLIC INFERRT_HAS_CUDA=1)
else()
    target_compile_definitions(inferrt_util PUBLIC INFERRT_HAS_CUDA=0)
endif()

# 只在 DEBUG 构建时暴露源代码信息（文件名、行号、调用语句）
target_compile_definitions(inferrt_util PUBLIC $<$<CONFIG:Debug>:IRT_EXPOSE_CODE=1>)
