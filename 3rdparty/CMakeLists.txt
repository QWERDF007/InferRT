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

# yaml-cpp -----------------------------
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "Disable yaml-cpp contrib" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "Disable yaml-cpp tools" FORCE)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "Disable yaml-cpp tests" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "Disable yaml-cpp install" FORCE)
add_subdirectory(yaml-cpp)

# vtest -----------------------------
# 父项目使用了 EXCLUDE_FROM_ALL，会导致该子目录中的所有目标不会被默认构建，
# 除非它们被其他目标依赖。
# set(OPENSSL_ROOT_DIR "E:/Software/OpenSSL-Win64")
# 
# CMake 的 FindOpenSSL.cmake 模块会在 Windows 注册表中查找 OpenSSL 的安装路径，具体是在：
# HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\OpenSSL (64-bit)_is1
# 这是 Inno Setup 安装程序（OpenSSL Windows 安装包使用的安装器）在注册表中记录的安装路径。
if(${PROJECT_NAME_UPPER}_BUILD_TESTS)
    add_subdirectory(vtest)
endif()

# benchmark
if (${PROJECT_NAME_UPPER}_BUILD_BENCHMARK)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing." FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark install to avoid overwriting vendor install." FORCE)
    set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "Disable benchmark warnings as errors." FORCE)
    set(BENCHMARK_FORCE_WERROR OFF CACHE BOOL "Disable forced benchmark warnings as errors." FORCE)
    add_subdirectory(benchmark)
endif()

# pybind11 - Python 绑定库
if(${PROJECT_NAME_UPPER}_BUILD_PYTHON)
    # 配置 pybind11 选项
    set(PYBIND11_INSTALL OFF CACHE BOOL "Disable pybind11 install" FORCE)
    set(PYBIND11_TEST OFF CACHE BOOL "Disable pybind11 tests" FORCE)
    set(PYBIND11_FINDPYTHON ON CACHE BOOL "Use FindPython for pybind11" FORCE)
    
    # 添加 pybind11 子目录
    add_subdirectory(pybind11)

    # dlpack - DLPack C API（INTERFACE 库，供 Python 绑定使用）
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dlpack/CMakeLists.txt")
        message(FATAL_ERROR
            "dlpack submodule not found at 3rdparty/dlpack. "
            "Run: git submodule update --init 3rdparty/dlpack")
    endif()
    set(BUILD_DOCS OFF CACHE BOOL "Disable dlpack docs" FORCE)
    set(BUILD_MOCK OFF CACHE BOOL "Disable dlpack mock executable" FORCE)
    add_subdirectory(dlpack)
endif()


# cpuinfo -----------------------------
# 仅构建 cpuinfo 库，避免其默认测试、工具和 benchmark 触发额外依赖下载。
set(CPUINFO_BUILD_TOOLS OFF CACHE BOOL "Disable cpuinfo tools" FORCE)
set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "Disable cpuinfo unit tests" FORCE)
set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "Disable cpuinfo mock tests" FORCE)
set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "Disable cpuinfo benchmarks" FORCE)
set(CPUINFO_BUILD_PKG_CONFIG OFF CACHE BOOL "Disable cpuinfo pkg-config manifest" FORCE)
add_subdirectory(cpuinfo EXCLUDE_FROM_ALL)
