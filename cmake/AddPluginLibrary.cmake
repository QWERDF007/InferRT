# 添加插件库的通用函数
# 参数:
#   PLUGIN_NAME - 插件名称
#   PRIVATE_LIBS - PRIVATE 链接的库列表
#   PUBLIC_LIBS - PUBLIC 链接的库列表
function(add_plugin_library PLUGIN_NAME)
    # 解析函数参数
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs PRIVATE_LIBS PUBLIC_LIBS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    set(TARGET_NAME "${PROJECT_NAME_LOWER}_${PLUGIN_NAME}")

    # 获取所有源文件的相对路径
    file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp *.cu)

    # 获取所有头文件的相对路径
    file(GLOB_RECURSE HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h *.hpp *.cuh)

    add_library(${TARGET_NAME} SHARED ${SOURCES} ${HEADERS})

    # 链接库
    target_link_libraries(${TARGET_NAME}
        PRIVATE
            ${ARG_PRIVATE_LIBS}
        PUBLIC
            ${ARG_PUBLIC_LIBS}
    )

    target_include_directories(${TARGET_NAME} 
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PLUGIN_NAME}> 
            $<INSTALL_INTERFACE:include>
    )

    configure_version(${TARGET_NAME} INFERRT ${PROJECT_NAME_LOWER}/${PLUGIN_NAME} ${PROJECT_VERSION})

    # 将名称转换为大写
    string(TOUPPER ${TARGET_NAME} TARGET_NAME_UPPER)
    # 设置额外编译定义
    set(TARGET_EXPORTS "${TARGET_NAME_UPPER}_BUILD_SHARED_LIBS")
    # 添加一个宏定义, 配合动态库导出函数、类
    # https://stackoverflow.com/a/67923443
    target_compile_definitions(${TARGET_NAME} PRIVATE ${TARGET_EXPORTS})

    # 添加接口头文件，链接目标后可以include, 无需另外包含头文件目录
    set(PLUGIN_HEADER "${PROJECT_NAME_LOWER}_${PLUGIN_NAME}_header")
    add_library(${PLUGIN_HEADER} INTERFACE)
    target_include_directories(${PLUGIN_HEADER}
        INTERFACE 
            ${CMAKE_CURRENT_SOURCE_DIR}/include 
            ${CMAKE_CURRENT_BINARY_DIR}/include
    )

    target_link_libraries(${TARGET_NAME} PUBLIC ${PLUGIN_HEADER})

    install(
        DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME_LOWER}/${PLUGIN_NAME}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME_LOWER}/${PLUGIN_NAME}
        COMPONENT dev
        PATTERN "detail" EXCLUDE
    )

    install(
        TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endfunction()
