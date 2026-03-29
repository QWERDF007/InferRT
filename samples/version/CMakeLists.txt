set(PLUGIN_NAME "version")
set(TARGET_NAME "${PROJECT_NAME_LOWER}_sample_${PLUGIN_NAME}")

# 获取所有源文件的相对路径
file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
# 获取所有头文件的相对路径
file(GLOB_RECURSE HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h *.hpp)

add_executable(${TARGET_NAME} ${SOURCES} ${HEADERS})


target_link_libraries(${TARGET_NAME}
    PRIVATE
        ${PROJECT_NAME_LOWER}_core ${PROJECT_NAME_LOWER}_core_header
    PUBLIC
)

install(
    TARGETS ${TARGET_NAME}
    RUNTIME DESTINATION samples
)
