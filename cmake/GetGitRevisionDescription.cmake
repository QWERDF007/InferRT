# - Returns a version string from Git
#
# 这些函数会在每次 Git 提交后强制触发一次重新配置（re-configure），从而确保你在构建系统中
# 使用到的变量值是可信的、最新的。
#
#  get_git_head_revision(<refspecvar> <hashvar> [<additional arguments to git describe> ...])
#
# 返回当前 HEAD 修订的 refspec（引用/分支）以及 sha 哈希。
#
#  git_describe(<var> [<additional arguments to git describe> ...])
#
# 返回对源代码树执行 git describe 的结果；如果发生错误，会调整输出使其在 CMake 中测试为 false。
#
#  git_get_exact_tag(<var> [<additional arguments to git describe> ...])
#
# 返回对源代码树执行 git describe --exact-match 的结果；如果没有精确匹配的 tag，会调整输出
# 使其在 CMake 中测试为 false。
#
#  git_local_changes(<var>)
#
# 针对未提交的修改，返回 "CLEAN" 或 "DIRTY"。
# 使用 "git diff-index --quiet HEAD --" 的返回码。
# 不考虑未跟踪（untracked）的文件。
#
# Requires CMake 2.6 or newer (uses the 'function' command)
#
# Original Author:
# 2009-2010 Ryan Pavlik <rpavlik@iastate.edu> <abiryan@ryand.net>
# http://academic.cleardefinition.com
# Iowa State University HCI Graduate Program/VRAC
#
# Copyright Iowa State University 2009-2010.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt)
 
if(__get_git_revision_description)
	# 防止重复 include 本模块
	return()
endif()
set(__get_git_revision_description YES)

# 必须在 include 时执行，而不是在函数调用时执行：获取本模块文件所在目录（用于定位 .in 模板文件）
get_filename_component(_gitdescmoddir ${CMAKE_CURRENT_LIST_FILE} PATH)


function(get_git_head_revision _refspecvar _branchvar _hashvar)
	# 返回当前源码树 Git 仓库的两个信息：
	# - refspec（HEAD 指向的引用，例如 refs/heads/main 或 refs/tags/v1.0.0）
	# - commit hash（HEAD 对应的提交哈希）
	#
	# 注意：这个函数不直接调用 git 命令。
	# 它通过读取 .git/HEAD 以及 refs 文件来得到 HEAD_REF / HEAD_HASH，并通过把这些文件
	# copy 到 build 目录的 "git-data" 中，来让 Git 状态变化时触发 CMake 重新配置。
	# 从当前源码目录开始向上查找 .git（支持在子目录中调用）
	set(GIT_PARENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
	set(GIT_DIR "${GIT_PARENT_DIR}/.git")
	while(NOT EXISTS "${GIT_DIR}")	# .git 目录没找到, 搜索父目录
		set(GIT_PREVIOUS_PARENT "${GIT_PARENT_DIR}")
		get_filename_component(GIT_PARENT_DIR ${GIT_PARENT_DIR} PATH)
		if(GIT_PARENT_DIR STREQUAL GIT_PREVIOUS_PARENT)
			# We have reached the root directory, we are not in git
			set(${_refspecvar} "GITDIR-NOTFOUND" PARENT_SCOPE)
			set(${_branchvar} "GITDIR-NOTFOUND" PARENT_SCOPE)
			set(${_hashvar} "GITDIR-NOTFOUND" PARENT_SCOPE)
			return()
		endif()
		set(GIT_DIR "${GIT_PARENT_DIR}/.git")
	endwhile()
	# 兼容 submodule：此时 .git 可能是一个文件，内容为 "gitdir: <path>"
	# 例如 submodule 的工作区里 .git 通常不是目录，而是一个指向真实 git dir 的文本文件。
	if(NOT IS_DIRECTORY ${GIT_DIR})
		file(READ ${GIT_DIR} submodule)
		string(REGEX REPLACE "gitdir: (.*)\n$" "\\1" GIT_DIR_RELATIVE ${submodule})
		get_filename_component(SUBMODULE_DIR ${GIT_DIR} PATH)
		get_filename_component(GIT_DIR ${SUBMODULE_DIR}/${GIT_DIR_RELATIVE} ABSOLUTE)
	endif()
	# 在 build 目录下创建缓存区：通过 configure_file 复制 HEAD/refs 等文件，使其变化能触发 CMake 重新配置
	set(GIT_DATA "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/git-data")
	if(NOT EXISTS "${GIT_DATA}")
		file(MAKE_DIRECTORY "${GIT_DATA}")
	endif()

	if(NOT EXISTS "${GIT_DIR}/HEAD")
		set(${_branchvar} "HEAD-NOTFOUND" PARENT_SCOPE)
		return()
	endif()
	set(HEAD_FILE "${GIT_DATA}/HEAD")
	# 将 .git/HEAD 复制到 build 目录，这样当 HEAD 内容改变（切换分支/checkout/commit）时
	# CMake 能检测到输入文件变化并触发重新配置。
	configure_file("${GIT_DIR}/HEAD" "${HEAD_FILE}" COPYONLY)

	# 使用 .in 模板生成一个临时 CMake 脚本，用于解析 HEAD 并得到 HEAD_REF / HEAD_HASH
	# grabRef.cmake 会进一步读取 refs/heads/* 或 packed-refs，并同样通过 configure_file
	# 把相关文件拷贝到 git-data，从而把这些文件纳入 CMake 的依赖追踪。
	configure_file("${_gitdescmoddir}/GetGitRevisionDescription.cmake.in"
		"${GIT_DATA}/grabRef.cmake"
		@ONLY)
	include("${GIT_DATA}/grabRef.cmake")

	# 从 HEAD_REF 推导出分支名。
	# HEAD_REF 通常为 refs/heads/<branch>；若为 tag 则为 refs/tags/<tag>。
	set(HEAD_BRANCH "UNKNOWN")
	if(HEAD_REF MATCHES "^refs/heads/(.+)$")
		set(HEAD_BRANCH "${CMAKE_MATCH_1}")
	elseif(HEAD_REF MATCHES "^refs/tags/(.+)$")
		set(HEAD_BRANCH "tag/${CMAKE_MATCH_1}")
	elseif(HEAD_REF)
		set(HEAD_BRANCH "${HEAD_REF}")
	endif()

	# 将结果返回给调用者
	set(${_refspecvar} "${HEAD_REF}" PARENT_SCOPE)
	set(${_branchvar} "${HEAD_BRANCH}" PARENT_SCOPE)
	set(${_hashvar} "${HEAD_HASH}" PARENT_SCOPE)
endfunction()

function(git_describe _var)
	# 调用 git describe 生成描述字符串（例如 tag-距离-g<hash>）；失败时输出会带 NOTFOUND
	if(NOT GIT_FOUND)
		find_package(Git QUIET)
	endif()
	get_git_head_revision(refspec branch hash)
	if(NOT GIT_FOUND)
		set(${_var} "GIT-NOTFOUND" PARENT_SCOPE)
		return()
	endif()
	if(NOT hash)
		set(${_var} "HEAD-HASH-NOTFOUND" PARENT_SCOPE)
		return()
	endif()

	# TODO sanitize
	#if((${ARGN}" MATCHES "&&") OR
	#	(ARGN MATCHES "||") OR
	#	(ARGN MATCHES "\\;"))
	#	message("Please report the following error to the project!")
	#	message(FATAL_ERROR "Looks like someone's doing something nefarious with git_describe! Passed arguments ${ARGN}")
	#endif()

	#message(STATUS "Arguments to execute_process: ${ARGN}")

	execute_process(COMMAND
		"${GIT_EXECUTABLE}"
		describe
		${hash}
		${ARGN}
		WORKING_DIRECTORY
		"${CMAKE_CURRENT_SOURCE_DIR}"
		RESULT_VARIABLE
		res
		OUTPUT_VARIABLE
		out
		ERROR_QUIET
		OUTPUT_STRIP_TRAILING_WHITESPACE)
	if(NOT res EQUAL 0)
		set(out "${out}-${res}-NOTFOUND")
	endif()

	set(${_var} "${out}" PARENT_SCOPE)
endfunction()

function(git_get_exact_tag _var)
	# 仅当当前提交精确匹配某个 tag 时才返回该 tag（否则返回 NOTFOUND 风格字符串）
	git_describe(out --exact-match ${ARGN})
	set(${_var} "${out}" PARENT_SCOPE)
endfunction()

function(git_local_changes _var)
	# 判断工作区是否有未提交修改：DIRTY 表示有修改；不考虑 untracked 文件
	if(NOT GIT_FOUND)
		find_package(Git QUIET)
	endif()
	get_git_head_revision(refspec branch hash)
	if(NOT GIT_FOUND)
		set(${_var} "GIT-NOTFOUND" PARENT_SCOPE)
		return()
	endif()
	if(NOT hash)
		set(${_var} "HEAD-HASH-NOTFOUND" PARENT_SCOPE)
		return()
	endif()

        execute_process(
                COMMAND "${GIT_EXECUTABLE}" update-index --refresh --quiet
                COMMAND "${GIT_EXECUTABLE}" diff-index --quiet HEAD --
		WORKING_DIRECTORY
		"${CMAKE_CURRENT_SOURCE_DIR}"
		RESULT_VARIABLE
		res
		OUTPUT_VARIABLE
		out
		ERROR_QUIET
		OUTPUT_STRIP_TRAILING_WHITESPACE)
	if(res EQUAL 0)
                set(${_var} "${_var}-NOTFOUND" PARENT_SCOPE)
	else()
		set(${_var} "DIRTY" PARENT_SCOPE)
	endif()
endfunction()

function(git_branch _var)
	# 返回当前分支名（detached HEAD 时通常为 "HEAD"）
	if(NOT GIT_FOUND)
		find_package(Git QUIET)
	endif()
	get_git_head_revision(refspec branch hash)
	if(NOT GIT_FOUND)
		set(${_var} "GIT-NOTFOUND" PARENT_SCOPE)
		return()
	endif()
	if(NOT hash)
		set(${_var} "HEAD-HASH-NOTFOUND" PARENT_SCOPE)
		return()
	endif()

	execute_process(COMMAND
		"${GIT_EXECUTABLE}"
                rev-parse --abbrev-ref --quiet HEAD
		WORKING_DIRECTORY
		"${CMAKE_CURRENT_SOURCE_DIR}"
		RESULT_VARIABLE
		res
		OUTPUT_VARIABLE
		out
		ERROR_QUIET
		OUTPUT_STRIP_TRAILING_WHITESPACE)

	if(NOT res EQUAL 0)
		set(out "${_var}-NOTFOUND")
	endif()

	set(${_var} "${out}" PARENT_SCOPE)
endfunction()
