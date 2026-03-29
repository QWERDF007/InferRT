# 必须在 include 时执行（而不是在函数调用时），以便获取本模块文件所在路径
get_filename_component(config_version_script_path ${CMAKE_CURRENT_LIST_FILE} PATH)

include(GetGitRevisionDescription)
# 获取当前仓库的 HEAD 引用（分支 refspec）以及提交哈希（commit）
# 得到 GIT_REFSPEC（当前 HEAD 的分支名或 tag），通常是, refs/heads/<branch>；若为 tag 则为 refs/tags/<tag>。并把它变成 CMake 变量
# 得到 REPO_BRANCH（当前 HEAD 的分支名）
# 得到 REPO_COMMIT（当前 HEAD 的 commit hash）
get_git_head_revision(GIT_REFSPEC REPO_BRANCH REPO_COMMIT)

# 拼接工程版本号与可选后缀（例如 -rc1/-dev 等）
set(PROJECT_VERSION "${PROJECT_VERSION}${PROJECT_VERSION_SUFFIX}")

function(configure_version target LIBPREFIX incpath VERSION_FULL)
    # 为指定 target 配置版本信息，并生成/安装版本相关头文件。
    #
    # 入参说明：
    # - target：要应用版本配置的 CMake target（库/可执行文件）
    # - LIBPREFIX：用于生成/缓存变量名的前缀（例如 <LIBPREFIX>_VERSION_MAJOR 等）
    # - incpath：生成的头文件相对 include 根目录的路径（会写到 build/include/<incpath>/...）
    # - VERSION_FULL：完整版本字符串（例如 1.2.3、1.2.3-rc1、1.2.3.4 等）
    #
    # 主要职责：
    # - 解析 VERSION_FULL 得到 major/minor/patch/tweak 以及可选后缀
    # - 计算 API 版本编码（major*100 + minor）以及构建版本字符串
    # - 根据模板生成版本头文件到 build/include，并添加到 target 的 include 路径
    # - 将版本信息以 CACHE INTERNAL 形式缓存，便于工程其他模块/脚本查询
    # - 安装生成的版本头文件（通常属于 dev 组件）
    string(TOUPPER "${target}" TARGET)

    # 提取版本后缀（以 '-' 开头的部分，例如 1.2.3-rc1 -> rc1）
    string(REGEX MATCH "-(.*)$" version_suffix "${VERSION_FULL}")
    set(VERSION_SUFFIX ${CMAKE_MATCH_1})

    # 提取版本数字组件（major/minor/patch[/tweak]）
    string(REGEX MATCHALL "[0-9]+" version_list "${VERSION_FULL}")
    list(GET version_list 0 VERSION_MAJOR)
    list(GET version_list 1 VERSION_MINOR)
    list(GET version_list 2 VERSION_PATCH)

    list(LENGTH version_list num_version_components)

    # 允许 3 段或 4 段版本号
    if(num_version_components EQUAL 3)
        set(VERSION_TWEAK 0)
    elseif(num_version_components EQUAL 4)
        list(GET version_list 3 VERSION_TWEAK)
    else()
        message(FATAL_ERROR "Version must have either 3 or 4 components")
    endif()

    # API 版本编码：major*100 + minor
    math(EXPR VERSION_API_CODE "${VERSION_MAJOR}*100 + ${VERSION_MINOR}")

    string(REPLACE "-" "_" tmp ${VERSION_FULL})
    # 构建版本字符串：<版本号>-<构建后缀>
    set(VERSION_BUILD "${tmp}-${CVCUDA_BUILD_SUFFIX}")

    set(VERSIONDEF_TEMPLATE "${config_version_script_path}/VersionDef.h.in")
    set(VERSIONDEF_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/include/${incpath}/VersionDef.h")

    # 由模板生成版本头文件（写入 build 目录），供编译与安装使用
    # - VersionDef.h：导出各版本数值/字符串常量
    # - VersionUtils.h：版本相关的辅助工具（如有）
    # 注意：为了让 BUILD_TIME 在每次 build 时都刷新，VersionDef.h 不在 configure 阶段生成。
    #       这里通过一个 ALL 的自定义 target，在构建阶段运行 cmake -P 脚本生成 VersionDef.h。
    #       该脚本内部会 string(TIMESTAMP ...) 获取当前时间，并 configure_file() 写出头文件。
    add_custom_target(generate_${target}_versiondef ALL
        COMMAND ${CMAKE_COMMAND}
            # 使用 :PATH/:STRING 显式指定类型，并避免把路径用引号包进变量值（Windows/MSBuild 下容易出错）
            -DINPUT:PATH=${VERSIONDEF_TEMPLATE}
            -DOUTPUT:PATH=${VERSIONDEF_OUTPUT}
            -DLIBPREFIX:STRING=${LIBPREFIX}
            -DVERSION_MAJOR:STRING=${VERSION_MAJOR}
            -DVERSION_MINOR:STRING=${VERSION_MINOR}
            -DVERSION_PATCH:STRING=${VERSION_PATCH}
            -DVERSION_TWEAK:STRING=${VERSION_TWEAK}
            -DVERSION_SUFFIX:STRING=${VERSION_SUFFIX}
            -DVERSION_FULL:STRING=${VERSION_FULL}
            -P "${config_version_script_path}/GenerateVersionDef.cmake"
        # 声明副产物，便于生成器（如 Ninja）追踪该步骤会产出哪个文件
        BYPRODUCTS "${VERSIONDEF_OUTPUT}"
        VERBATIM
    )
    # 确保在编译/链接 ${target} 前先生成 VersionDef.h
    add_dependencies(${target} generate_${target}_versiondef)
    configure_file(${config_version_script_path}/VersionUtils.h.in include/${incpath}/detail/VersionUtils.h @ONLY ESCAPE_QUOTES)

    # 将版本信息缓存为 INTERNAL，便于工程其他位置引用（不会显示在常规 cache GUI 中）
    set(${LIBPREFIX}_VERSION_FULL ${VERSION_FULL} CACHE INTERNAL "${TARGET} full version")
    set(${LIBPREFIX}_VERSION_MAJOR ${VERSION_MAJOR} CACHE INTERNAL "${TARGET} major version")
    set(${LIBPREFIX}_VERSION_MINOR ${VERSION_MINOR} CACHE INTERNAL "${TARGET} minor version")
    set(${LIBPREFIX}_VERSION_PATCH ${VERSION_PATCH} CACHE INTERNAL "${TARGET} patch version")
    set(${LIBPREFIX}_VERSION_TWEAK ${VERSION_TWEAK} CACHE INTERNAL "${TARGET} tweak version")
    set(${LIBPREFIX}_VERSION_SUFFIX ${VERSION_SUFFIX} CACHE INTERNAL "${TARGET} version suffix")
    set(${LIBPREFIX}_VERSION_API ${VERSION_MAJOR}.${VERSION_MINOR} CACHE INTERNAL "${TARGET} API version")
    set(${LIBPREFIX}_VERSION_API_CODE ${VERSION_API_CODE} CACHE INTERNAL "${TARGET} API code")
    set(${LIBPREFIX}_VERSION_BUILD ${VERSION_BUILD} CACHE INTERNAL "${TARGET} build version")

    # 让 target 在构建时能找到生成的头文件（build/include）
    target_include_directories(${target}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    )

    # 安装生成的头文件（通常用于 dev 组件）
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/${incpath}/VersionDef.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${incpath}
            COMPONENT dev)
    # install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/${incpath}/detail/VersionUtils.h
    #         DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${incpath}/detail
    #         COMPONENT dev)
endfunction()

function(configure_symbol_versioning dso_target VERPREFIX input_targets)
    # Create exports file for symbol versioning ---------------------------------
    # 为共享库生成符号版本脚本（linker version-script），用于控制导出符号/ABI
    set(EXPORTS_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/exports.ldscript")
    target_link_libraries(${dso_target}
        PRIVATE
        -Wl,--version-script ${EXPORTS_OUTPUT}
    )
    set(ALL_SOURCES "")
    foreach(tgt ${input_targets})
        get_target_property(tgt_sources ${tgt} SOURCES)
        get_target_property(tgt_srcdir ${tgt} SOURCE_DIR)

        foreach(src ${tgt_sources})
            if(${src} MATCHES "^/") # absolute paths?
                # 绝对路径直接使用
                list(APPEND ALL_SOURCES ${src})
            else()
                # 相对路径补全为源目录下的绝对路径
                list(APPEND ALL_SOURCES ${tgt_srcdir}/${src})
            endif()
        endforeach()
    endforeach()

    set(GEN_EXPORTS_SCRIPT "${config_version_script_path}/CreateExportsFile.cmake")

    # 通过 CMake 脚本扫描源文件并生成 exports.ldscript
    add_custom_command(OUTPUT ${EXPORTS_OUTPUT}
        COMMAND ${CMAKE_COMMAND} -DSOURCES="${ALL_SOURCES}"
                                 -DVERPREFIX=${VERPREFIX}
                                 -DOUTPUT=${EXPORTS_OUTPUT}
                                 -P "${GEN_EXPORTS_SCRIPT}"
        DEPENDS ${GEN_EXPORTS_SCRIPT} ${ALL_SOURCES})

    # 确保链接共享库前先生成版本脚本
    add_custom_target(create_${dso_target}_exports_file DEPENDS ${EXPORTS_OUTPUT})
    add_dependencies(${dso_target} create_${dso_target}_exports_file)
endfunction()
