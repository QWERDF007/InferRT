# ConfigPython.cmake
# InferRT Python 绑定的环境配置

include("${CMAKE_CURRENT_LIST_DIR}/ConfigDependencyDefaults.cmake")

# Resolve one canonical root before populating FindPython's cache variables.
# Legacy FindPython cache entries are only hints; explicit current -D values,
# parent variables and the project manifest are resolved in that order.
inferrt_dependency_variable_is_explicit(
    INFERRT_PYTHON_EXE INFERRT_DEPENDENCY_PYTHON_EXE _inferrt_python_exe_explicit)
inferrt_dependency_variable_is_explicit(
    Python_EXECUTABLE INFERRT_DEPENDENCY_PYTHON_EXECUTABLE _inferrt_python_executable_explicit)
inferrt_dependency_variable_is_explicit(
    Python_ROOT_DIR INFERRT_DEPENDENCY_PYTHON_ROOT _inferrt_python_root_explicit)
inferrt_dependency_variable_is_explicit(
    INFERRT_PYTHON_ROOT INFERRT_DEPENDENCY_PYTHON_ROOT_HINT _inferrt_python_root_hint_explicit)
inferrt_dependency_variable_is_explicit(
    Python3_ROOT_DIR INFERRT_DEPENDENCY_PYTHON3_ROOT _inferrt_python3_root_explicit)
inferrt_dependency_variable_is_explicit(
    Python3_EXECUTABLE INFERRT_DEPENDENCY_PYTHON3_EXECUTABLE _inferrt_python3_executable_explicit)

# Python3_* is commonly populated as a normal compatibility hint by a parent
# project.  Keep that legacy hint below the manifest default; a cache entry
# (including a command-line -D entry) remains eligible as an explicit choice.
get_property(_inferrt_python3_root_cached CACHE Python3_ROOT_DIR PROPERTY TYPE SET)
get_property(_inferrt_python3_executable_cached CACHE Python3_EXECUTABLE PROPERTY TYPE SET)
if(NOT _inferrt_python3_root_cached)
    set(_inferrt_python3_root_explicit OFF)
endif()
if(NOT _inferrt_python3_executable_cached)
    set(_inferrt_python3_executable_explicit OFF)
endif()

if(WIN32)
    inferrt_dependency_default(python-torch-zlib _inferrt_python_default_root)
endif()

set(_inferrt_python_root)
set(_inferrt_python_source)
set(_inferrt_python_executable_hint)
if(_inferrt_python_exe_explicit AND NOT INFERRT_PYTHON_EXE STREQUAL "")
    get_filename_component(_inferrt_python_root "${INFERRT_PYTHON_EXE}" DIRECTORY)
    set(_inferrt_python_source "INFERRT_PYTHON_EXE")
    set(_inferrt_python_executable_hint "${INFERRT_PYTHON_EXE}")
elseif(_inferrt_python_executable_explicit AND NOT Python_EXECUTABLE STREQUAL "")
    get_filename_component(_inferrt_python_root "${Python_EXECUTABLE}" DIRECTORY)
    set(_inferrt_python_source "Python_EXECUTABLE")
    set(_inferrt_python_executable_hint "${Python_EXECUTABLE}")
elseif(_inferrt_python_root_explicit AND NOT Python_ROOT_DIR STREQUAL "")
    set(_inferrt_python_root "${Python_ROOT_DIR}")
    set(_inferrt_python_source "Python_ROOT_DIR")
elseif(_inferrt_python_root_hint_explicit AND NOT INFERRT_PYTHON_ROOT STREQUAL "")
    set(_inferrt_python_root "${INFERRT_PYTHON_ROOT}")
    set(_inferrt_python_source "INFERRT_PYTHON_ROOT")
elseif(_inferrt_python3_root_explicit AND NOT Python3_ROOT_DIR STREQUAL "")
    set(_inferrt_python_root "${Python3_ROOT_DIR}")
    set(_inferrt_python_source "Python3_ROOT_DIR")
elseif(_inferrt_python3_executable_explicit AND NOT Python3_EXECUTABLE STREQUAL "")
    get_filename_component(_inferrt_python_root "${Python3_EXECUTABLE}" DIRECTORY)
    set(_inferrt_python_source "Python3_EXECUTABLE")
    set(_inferrt_python_executable_hint "${Python3_EXECUTABLE}")
elseif(DEFINED ENV{INFERRT_PYTHON_EXE} AND NOT "$ENV{INFERRT_PYTHON_EXE}" STREQUAL "")
    get_filename_component(_inferrt_python_root "$ENV{INFERRT_PYTHON_EXE}" DIRECTORY)
    set(_inferrt_python_source "environment")
elseif(DEFINED ENV{Python_ROOT_DIR} AND NOT "$ENV{Python_ROOT_DIR}" STREQUAL "")
    set(_inferrt_python_root "$ENV{Python_ROOT_DIR}")
    set(_inferrt_python_source "environment")
elseif(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
    set(_inferrt_python_root "$ENV{CONDA_PREFIX}")
    set(_inferrt_python_source "environment")
elseif(_inferrt_python_default_root AND EXISTS "${_inferrt_python_default_root}")
    set(_inferrt_python_root "${_inferrt_python_default_root}")
    set(_inferrt_python_source "project-default")
elseif(DEFINED Python3_ROOT_DIR AND NOT Python3_ROOT_DIR STREQUAL "")
    set(_inferrt_python_root "${Python3_ROOT_DIR}")
    set(_inferrt_python_source "legacy-cache")
elseif(DEFINED Python3_EXECUTABLE AND NOT Python3_EXECUTABLE STREQUAL "")
    get_filename_component(_inferrt_python_root "${Python3_EXECUTABLE}" DIRECTORY)
    set(_inferrt_python_source "legacy-cache")
endif()

if(_inferrt_python_root)
    if(_inferrt_python_source STREQUAL "project-default" OR _inferrt_python_source STREQUAL "legacy-cache")
        inferrt_dependency_cache_set(
            INFERRT_PYTHON_ROOT "${_inferrt_python_root}" PATH
            "InferRT binding preferred Python environment"
            INFERRT_DEPENDENCY_PYTHON_ROOT_HINT project-default)
        set(_inferrt_python_source "project-default")
    endif()

    if(NOT DEFINED Python_ROOT_DIR OR NOT _inferrt_python_root_explicit)
        inferrt_dependency_cache_set(
            Python_ROOT_DIR "${_inferrt_python_root}" PATH
            "Python root directory"
            INFERRT_DEPENDENCY_PYTHON_ROOT ${_inferrt_python_source})
    endif()

    if(_inferrt_python_executable_hint)
        set(_inferrt_python_executable "${_inferrt_python_executable_hint}")
    elseif(WIN32)
        set(_inferrt_python_executable "${_inferrt_python_root}/python.exe")
    else()
        set(_inferrt_python_executable "${_inferrt_python_root}/bin/python")
    endif()

    if(_inferrt_python_executable_hint)
        inferrt_dependency_cache_set(
            Python_EXECUTABLE "${_inferrt_python_executable}" FILEPATH
            "Python interpreter"
            INFERRT_DEPENDENCY_PYTHON_EXECUTABLE user)
    elseif(_inferrt_python_executable_explicit)
        inferrt_dependency_record_value(
            INFERRT_DEPENDENCY_PYTHON_EXECUTABLE "${Python_EXECUTABLE}" user)
    elseif(EXISTS "${_inferrt_python_executable}")
        inferrt_dependency_cache_set(
            Python_EXECUTABLE "${_inferrt_python_executable}" FILEPATH
            "Python interpreter"
            INFERRT_DEPENDENCY_PYTHON_EXECUTABLE ${_inferrt_python_source})
    endif()

    if(_inferrt_python3_root_explicit OR _inferrt_python3_executable_explicit)
        # Preserve explicit legacy FindPython3 variables for callers that use
        # that module in the same configure pass.
        if(_inferrt_python3_root_explicit AND DEFINED Python3_ROOT_DIR)
            inferrt_dependency_record_value(
                INFERRT_DEPENDENCY_PYTHON3_ROOT "${Python3_ROOT_DIR}" user)
        endif()
        if(_inferrt_python3_executable_explicit AND DEFINED Python3_EXECUTABLE)
            inferrt_dependency_record_value(
                INFERRT_DEPENDENCY_PYTHON3_EXECUTABLE "${Python3_EXECUTABLE}" user)
        endif()
    endif()
endif()

unset(_inferrt_python_executable)
unset(_inferrt_python_root)
unset(_inferrt_python_source)
unset(_inferrt_python_executable_hint)
unset(_inferrt_python_default_root)
unset(_inferrt_python_exe_explicit)
unset(_inferrt_python_executable_explicit)
unset(_inferrt_python_root_explicit)
unset(_inferrt_python_root_hint_explicit)
unset(_inferrt_python3_root_explicit)
unset(_inferrt_python3_executable_explicit)
unset(_inferrt_python3_root_cached)
unset(_inferrt_python3_executable_cached)

unset(_inferrt_python_root)
unset(_inferrt_python_default_root)

# 查找 Python 包
find_package(Python COMPONENTS Interpreter Development REQUIRED)
