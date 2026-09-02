# ConfigPython.cmake
# InferRT Python 绑定的环境配置

if(NOT DEFINED Python_ROOT_DIR AND NOT DEFINED Python_EXECUTABLE)
    if(DEFINED ENV{Python_ROOT_DIR})
        set(Python_ROOT_DIR "$ENV{Python_ROOT_DIR}")
    elseif(DEFINED ENV{CONDA_PREFIX})
        set(Python_ROOT_DIR "$ENV{CONDA_PREFIX}")
    elseif(DEFINED INFERRT_PYTHON_ROOT AND EXISTS "${INFERRT_PYTHON_ROOT}")
        set(Python_ROOT_DIR "${INFERRT_PYTHON_ROOT}")
    endif()
endif()

# 查找 Python 包
find_package(Python COMPONENTS Interpreter Development REQUIRED)
