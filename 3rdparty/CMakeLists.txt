set(CMAKE_FOLDER 3rdparty)

# GoogleTest -----------------------------
set(INSTALL_GTEST off)
set(BUILD_GMOCK on)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
if(INFERRT_BUILD_TESTS)
    add_subdirectory(googletest)
endif()

# cxxopts -----------------------------
set(CXXOPTS_BUILD_EXAMPLES OFF)
set(CXXOPTS_BUILD_TESTS OFF)
set(CXXOPTS_ENABLE_INSTALL OFF)
set(CXXOPTS_ENABLE_WARNINGS OFF)
add_subdirectory(cxxopts)

# yaml-cpp -----------------------------
set(YAML_CPP_BUILD_CONTRIB OFF)
set(YAML_CPP_BUILD_TOOLS OFF)
set(YAML_CPP_BUILD_TESTS OFF)
set(YAML_CPP_INSTALL OFF)
add_subdirectory(yaml-cpp)

# vtest -----------------------------
# 父项目使用了 EXCLUDE_FROM_ALL，会导致该子目录中的所有目标不会被默认构建，
# 除非它们被其他目标依赖。
if(${PROJECT_NAME_UPPER}_BUILD_TESTS)
    add_subdirectory(vtest)
endif()

# benchmark
if (${PROJECT_NAME_UPPER}_BUILD_BENCHMARK)
    set(BENCHMARK_ENABLE_TESTING OFF)
    set(BENCHMARK_ENABLE_INSTALL OFF)
    set(BENCHMARK_ENABLE_WERROR OFF)
    set(BENCHMARK_FORCE_WERROR OFF)
    add_subdirectory(benchmark)
endif()

# pybind11 - Python 绑定库
if(${PROJECT_NAME_UPPER}_BUILD_PYTHON)
    # 配置 pybind11 选项
    set(PYBIND11_INSTALL OFF)
    set(PYBIND11_TEST OFF)
    set(PYBIND11_FINDPYTHON ON)
    
    # 添加 pybind11 子目录
    add_subdirectory(pybind11)

    # dlpack - DLPack C API（INTERFACE 库，供 Python 绑定使用）
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dlpack/CMakeLists.txt")
        message(FATAL_ERROR
            "dlpack submodule not found at 3rdparty/dlpack. "
            "Run: git submodule update --init 3rdparty/dlpack")
    endif()
    set(BUILD_DOCS OFF)
    set(BUILD_MOCK OFF)
    add_subdirectory(dlpack)
endif()


# cpuinfo -----------------------------
# 仅构建 cpuinfo 库，避免其默认测试、工具和 benchmark 触发额外依赖下载。
set(CPUINFO_BUILD_TOOLS OFF)
set(CPUINFO_BUILD_UNIT_TESTS OFF)
set(CPUINFO_BUILD_MOCK_TESTS OFF)
set(CPUINFO_BUILD_BENCHMARKS OFF)
set(CPUINFO_BUILD_PKG_CONFIG OFF)
add_subdirectory(cpuinfo EXCLUDE_FROM_ALL)
