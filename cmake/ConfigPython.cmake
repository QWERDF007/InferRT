# ConfigPython.cmake
# InferRT Python 绑定的环境配置

# 设置首选的 Python 环境根目录
set(INFERRT_PYTHON_ROOT "D:/Software/anaconda3/envs/py312" CACHE PATH 
    "InferRT 绑定首选的 Python 环境")

# 如果 Python_ROOT_DIR 未设置且 INFERRT_PYTHON_ROOT 存在，则配置 Python_ROOT_DIR
if(NOT Python_ROOT_DIR AND EXISTS "${INFERRT_PYTHON_ROOT}")
    set(Python_ROOT_DIR "${INFERRT_PYTHON_ROOT}" CACHE PATH 
        "InferRT 绑定使用的 Python 根目录" FORCE)
endif()

# 如果 Python_EXECUTABLE 未设置且可执行文件存在，则配置 Python_EXECUTABLE
if(NOT Python_EXECUTABLE AND EXISTS "${INFERRT_PYTHON_ROOT}/python.exe")
    set(Python_EXECUTABLE "${INFERRT_PYTHON_ROOT}/python.exe" CACHE FILEPATH 
        "InferRT 绑定使用的 Python 可执行文件" FORCE)
endif()

# 查找 Python 包
find_package(Python COMPONENTS Interpreter Development REQUIRED)

