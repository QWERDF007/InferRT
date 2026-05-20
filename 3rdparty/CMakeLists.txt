set(CMAKE_FOLDER 3rdparty)

# GoogleTest -----------------------------
set(INSTALL_GTEST off)
set(BUILD_GMOCK on)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
if(INFERRT_BUILD_TESTS)
    add_subdirectory(googletest)
endif()

# cxxopts -----------------------------
set(CXXOPTS_BUILD_EXAMPLES OFF CACHE BOOL "Disable cxxopts examples" FORCE)
set(CXXOPTS_BUILD_TESTS OFF CACHE BOOL "Disable cxxopts tests" FORCE)
set(CXXOPTS_ENABLE_INSTALL OFF CACHE BOOL "Disable cxxopts install" FORCE)
set(CXXOPTS_ENABLE_WARNINGS OFF CACHE BOOL "Disable cxxopts warnings" FORCE)
add_subdirectory(cxxopts)

# vtest -----------------------------
# 父项目使用了 EXCLUDE_FROM_ALL，会导致该子目录中的所有目标不会被默认构建，
# 除非它们被其他目标依赖。
# set(OPENSSL_ROOT_DIR "E:/Software/OpenSSL-Win64")
# 
# CMake 的 FindOpenSSL.cmake 模块会在 Windows 注册表中查找 OpenSSL 的安装路径，具体是在：
# HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\OpenSSL (64-bit)_is1
# 这是 Inno Setup 安装程序（OpenSSL Windows 安装包使用的安装器）在注册表中记录的安装路径。
add_subdirectory(vtest)

# benchmark
if (${PROJECT_NAME_UPPER}_BUILD_BENCHMARK)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing.")
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark install to avoid overwriting vendor install.")
    add_subdirectory(benchmark)
endif()

# pybind11 - Python 绑定库
if(${PROJECT_NAME_UPPER}_BUILD_PYTHON)
    # 配置 Python 根目录（如果尚未设置）
    if(NOT Python_ROOT_DIR AND EXISTS "${INFERRT_PYTHON_ROOT}")
        set(Python_ROOT_DIR "${INFERRT_PYTHON_ROOT}" CACHE PATH "Python root used by InferRT bindings" FORCE)
    endif()
    
    # 配置 Python 可执行文件路径（如果尚未设置）
    if(NOT Python_EXECUTABLE AND EXISTS "${INFERRT_PYTHON_ROOT}/python.exe")
        set(Python_EXECUTABLE "${INFERRT_PYTHON_ROOT}/python.exe" CACHE FILEPATH
            "Python executable used by InferRT bindings" FORCE)
    endif()
    
    # 配置 pybind11 选项
    set(PYBIND11_INSTALL OFF CACHE BOOL "Disable pybind11 install" FORCE)
    set(PYBIND11_TEST OFF CACHE BOOL "Disable pybind11 tests" FORCE)
    set(PYBIND11_FINDPYTHON ON CACHE BOOL "Use FindPython for pybind11" FORCE)
    
    # 添加 pybind11 子目录
    add_subdirectory(pybind11)
endif()
