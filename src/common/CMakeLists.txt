set(PLUGIN_NAME "common")
set(TARGET_NAME "${PROJECT_NAME_LOWER}_${PLUGIN_NAME}")

# 获取所有源文件的相对路径
file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
# 获取所有头文件的相对路径
file(GLOB_RECURSE HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h *.hpp)


# 如果编译模式是Debug或RelWithDebInfo，则添加预处理定义QT_QML_DEBUG。用于开启Qt Quick调试功能
target_compile_definitions(${TARGET_NAME}
    PRIVATE $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:QT_QML_DEBUG>
)

target_link_libraries(${TARGET_NAME}
    PRIVATE
        
    PUBLIC        
)

target_include_directories(${TARGET_NAME} 
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PLUGIN_NAME}> # for QML_ELEMENT https://bugreports.qt.io/browse/QTBUG-87221
        $<INSTALL_INTERFACE:include>
)

# 将名称转换为大写
string(TOUPPER ${TARGET_NAME} TARGET_NAME_UPPER)
# 设置额外编译定义
set(TARGET_EXPORTS "${TARGET_NAME_UPPER}_BUILD_SHARED_LIBS")
# 添加一个宏定义, 配合动态库导出函数、类
# https://stackoverflow.com/a/67923443
target_compile_definitions(${TARGET_NAME} PRIVATE ${TARGET_EXPORTS})

# 添加接口头文件，链接目标后可以include, 无需另外包含头文件目录
set(PLUGIN_HEADER "${PROJECT_NAME_LOWER}_${PLUGIN_NAME}_header")
add_library(${PROJECT_NAME_LOWER}_${PLUGIN_NAME}_header INTERFACE)
target_include_directories(${PLUGIN_HEADER}
    INTERFACE 
        ${CMAKE_CURRENT_SOURCE_DIR}/include 
        ${CMAKE_CURRENT_BINARY_DIR}/include
)
