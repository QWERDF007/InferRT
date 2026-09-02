add_library(${PROJECT_NAME_LOWER}_sample_common INTERFACE)
target_link_libraries(${PROJECT_NAME_LOWER}_sample_common
    INTERFACE
        ${PROJECT_NAME_LOWER}_util
)
target_include_directories(${PROJECT_NAME_LOWER}_sample_common
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)

add_subdirectory(version)

if(${PROJECT_NAME_UPPER}_ENABLE_CUDA)
    add_subdirectory(cuda)
endif()
if(${PROJECT_NAME_UPPER}_ENABLE_CUDA AND ${PROJECT_NAME_UPPER}_BUILD_TENSORRT)
    add_subdirectory(features)
    add_subdirectory(model)
    add_subdirectory(engine)
endif()
