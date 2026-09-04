# 添加插件库的通用函数
# 参数:
#   PLUGIN_NAME - 插件名称
#   PRIVATE_LIBS - PRIVATE 链接的库列表
#   PUBLIC_LIBS - PUBLIC 链接的库列表
function(add_plugin_library PLUGIN_NAME)
    # 解析函数参数
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs PRIVATE_LIBS PUBLIC_LIBS PRIVATE_INCS PUBLIC_INCS EXCLUDE_SOURCES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    set(TARGET_NAME "${PROJECT_NAME_LOWER}_${PLUGIN_NAME}")
    set(TARGET_EXPORT_SET "${PROJECT_NAME}_${PLUGIN_NAME}Targets")

    # 获取所有源文件的相对路径
    file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp *.cu)

    if(ARG_EXCLUDE_SOURCES)
        list(REMOVE_ITEM SOURCES ${ARG_EXCLUDE_SOURCES})
    endif()

    # 获取所有头文件的相对路径
    file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h *.hpp *.cuh)

    add_library(${TARGET_NAME} SHARED ${SOURCES} ${HEADERS})
    add_library(${PROJECT_NAME}::${PLUGIN_NAME} ALIAS ${TARGET_NAME})

    inferrt_apply_compile_options(${TARGET_NAME})

    # Public headers are stored as UTF-8 (including documentation comments).
    # Propagate the source charset to consumers so MSVC does not parse an
    # installed header with the active system code page.
    if(MSVC)
        target_compile_options(${TARGET_NAME} INTERFACE
            "$<$<COMPILE_LANGUAGE:CXX>:/utf-8>"
        )
    endif()

    set_target_properties(${TARGET_NAME} PROPERTIES
        EXPORT_NAME ${PLUGIN_NAME}
    )

    # 链接库
    target_link_libraries(${TARGET_NAME}
        PRIVATE
            ${ARG_PRIVATE_LIBS}
        PUBLIC
            ${ARG_PUBLIC_LIBS}
    )

    target_include_directories(${TARGET_NAME} 
        PRIVATE
            ${ARG_PRIVATE_INCS}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PLUGIN_NAME}> 
            $<INSTALL_INTERFACE:include>
            ${ARG_PUBLIC_INCS}
    )

    # 将名称转换为大写
    string(TOUPPER ${TARGET_NAME} TARGET_NAME_UPPER)
    # 设置额外编译定义
    set(TARGET_EXPORTS "${TARGET_NAME_UPPER}_BUILD_SHARED_LIBS")
    # 添加一个宏定义, 配合动态库导出函数、类
    # https://stackoverflow.com/a/67923443
    target_compile_definitions(${TARGET_NAME} PRIVATE ${TARGET_EXPORTS})

    # 将插件名称转换为大写，用于生成 Export.h
    string(TOUPPER ${PLUGIN_NAME} PLUGIN_NAME_UPPER)
    set(LIBPREFIX ${PLUGIN_NAME_UPPER})
    configure_file(
        ${CMAKE_SOURCE_DIR}/cmake/Export.h.in
        ${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME_LOWER}/${PLUGIN_NAME}/Export.h
        @ONLY
    )


    # 添加接口头文件，链接目标后可以include, 无需另外包含头文件目录
    set(PLUGIN_HEADER "${PROJECT_NAME_LOWER}_${PLUGIN_NAME}_header")
    add_library(${PLUGIN_HEADER} INTERFACE)
    target_include_directories(${PLUGIN_HEADER}
        INTERFACE 
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )

    # DIRECTORY path/to/dir 会安装 dir 目录本身及其内容
    # DIRECTORY path/to/dir/ - 只安装 dir 目录的内容（不包含 dir 本身）
    install(
        DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME_LOWER}/${PLUGIN_NAME}/ 
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME_LOWER}/${PLUGIN_NAME} # e.g. <install-prefix>/include/inferrt/cvcuda
        COMPONENT dev
        FILES_MATCHING
            PATTERN "*.h"
            PATTERN "*.hh"
            PATTERN "*.hpp"
            PATTERN "*.hxx"
            PATTERN "*.cuh"
        PATTERN "detail" EXCLUDE
    )

    install(
        TARGETS ${TARGET_NAME}
        EXPORT ${TARGET_EXPORT_SET}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

    # 安装生成的头文件 Export.h（通常用于 dev 组件）
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME_LOWER}/${PLUGIN_NAME}/Export.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME_LOWER}/${PLUGIN_NAME}
            COMPONENT dev)
endfunction()
