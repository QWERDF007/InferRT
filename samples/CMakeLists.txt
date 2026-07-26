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
add_subdirectory(cuda)
add_subdirectory(features)
add_subdirectory(model)
add_subdirectory(engine)
