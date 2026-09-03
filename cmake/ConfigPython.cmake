# ConfigPython.cmake
# InferRT Python 绑定的环境配置

include("${CMAKE_CURRENT_LIST_DIR}/ConfigDependencyDefaults.cmake")

# Resolve one canonical root before populating FindPython's cache variables.
if(DEFINED Python_EXECUTABLE AND NOT Python_EXECUTABLE STREQUAL "")
    get_filename_component(_inferrt_python_root "${Python_EXECUTABLE}" DIRECTORY)
    if(NOT WIN32)
        get_filename_component(_inferrt_python_root "${_inferrt_python_root}" DIRECTORY)
    endif()
elseif(DEFINED Python_ROOT_DIR AND NOT Python_ROOT_DIR STREQUAL "")
    set(_inferrt_python_root "${Python_ROOT_DIR}")
elseif(DEFINED INFERRT_PYTHON_ROOT AND NOT INFERRT_PYTHON_ROOT STREQUAL "")
    set(_inferrt_python_root "${INFERRT_PYTHON_ROOT}")
elseif(DEFINED CACHE{Python3_ROOT_DIR} AND NOT Python3_ROOT_DIR STREQUAL "")
    set(_inferrt_python_root "${Python3_ROOT_DIR}")
elseif(DEFINED CACHE{Python3_EXECUTABLE} AND NOT Python3_EXECUTABLE STREQUAL "")
    get_filename_component(_inferrt_python_root "${Python3_EXECUTABLE}" DIRECTORY)
    if(NOT WIN32)
        get_filename_component(_inferrt_python_root "${_inferrt_python_root}" DIRECTORY)
    endif()
elseif(DEFINED ENV{Python_ROOT_DIR} AND NOT "$ENV{Python_ROOT_DIR}" STREQUAL "")
    set(_inferrt_python_root "$ENV{Python_ROOT_DIR}")
elseif(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
    set(_inferrt_python_root "$ENV{CONDA_PREFIX}")
elseif(WIN32)
    inferrt_dependency_default(python-torch-zlib _inferrt_python_default_root)
    if(_inferrt_python_default_root AND EXISTS "${_inferrt_python_default_root}")
        set(_inferrt_python_root "${_inferrt_python_default_root}")
    endif()
elseif(DEFINED Python3_ROOT_DIR AND NOT Python3_ROOT_DIR STREQUAL "")
    set(_inferrt_python_root "${Python3_ROOT_DIR}")
elseif(DEFINED Python3_EXECUTABLE AND NOT Python3_EXECUTABLE STREQUAL "")
    get_filename_component(_inferrt_python_root "${Python3_EXECUTABLE}" DIRECTORY)
    if(NOT WIN32)
        get_filename_component(_inferrt_python_root "${_inferrt_python_root}" DIRECTORY)
    endif()
endif()

if(_inferrt_python_root)
    set(INFERRT_PYTHON_ROOT "${_inferrt_python_root}" CACHE PATH
        "InferRT binding preferred Python environment")
endif()

if(NOT DEFINED Python_ROOT_DIR OR Python_ROOT_DIR STREQUAL "")
    if(_inferrt_python_root AND EXISTS "${_inferrt_python_root}")
        set(Python_ROOT_DIR "${_inferrt_python_root}" CACHE PATH
            "Python root directory")
    endif()
endif()

if(NOT DEFINED Python_EXECUTABLE OR Python_EXECUTABLE STREQUAL "")
    if(DEFINED _inferrt_python_root AND NOT _inferrt_python_root STREQUAL "")
        if(WIN32)
            set(_inferrt_python_executable "${_inferrt_python_root}/python.exe")
        else()
            set(_inferrt_python_executable "${_inferrt_python_root}/bin/python")
        endif()
        if(EXISTS "${_inferrt_python_executable}")
            set(Python_EXECUTABLE "${_inferrt_python_executable}" CACHE FILEPATH
                "Python interpreter")
        endif()
        unset(_inferrt_python_executable)
    elseif(DEFINED Python3_EXECUTABLE AND NOT Python3_EXECUTABLE STREQUAL "")
        set(Python_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH
            "Python interpreter")
    endif()
endif()

unset(_inferrt_python_root)
unset(_inferrt_python_default_root)

# 查找 Python 包
find_package(Python COMPONENTS Interpreter Development REQUIRED)
