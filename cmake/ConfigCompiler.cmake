
# 将 C++ 标准设置为 20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# 在 RelWithDebInfo 模式下给 C/C++ 编译器添加 O3 和 ggdb 参数
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3 -ggdb")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -O3 -ggdb")


if(WARNINGS_AS_ERRORS)
    # 设置 C 语言警告为错误
    set(C_WARNING_ERROR_FLAG "-Werror")
    # 设置 CUDA 语言警告全部为错误
    set(CUDA_WARNING_ERROR_FLAG "-Werror all-warnings")
endif()

# -Wall：启用所有警告
# -Wno-unknown-pragmas：禁止对未知的编译器指令发出警告
# -Wpointer-arith：对指针算术运算发出警告
# -Wmissing-declarations：对缺少声明的函数或变量发出警告
# -Wredundant-decls：对冗余的声明发出警告
# -Wmultichar：对多字符字符常量发出警告
# -Wno-unused-local-typedefs：禁止对未使用的局部类型定义发出警告
# -Wunused：对未使用的变量、函数或标签发出警告
# Match warning setup with GVS
if (MSVC)
    # set(C_WARNING_FLAGS "-Wall")
    # /EHa: 启用 C++ 异常处理和 SEH 异常（跨 DLL 异常传播所需）
    # /utf-8: 将源文件和执行字符集设置为 UTF-8
    # /wd4251: 关闭 STL 成员经 DLL 导出时的接口警告
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHa /utf-8 /bigobj /arch:AVX2")
    set(C_WARNING_FLAGS "-W4 /wd4251")
    # set(CXX_WARNING_FLAGS "/permissive-")
else ()
    # -mavx2: 启用 AVX2 指令集，供模板匹配等 SIMD 实现使用。
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
    set(C_WARNING_FLAGS "-Wall -Wno-unknown-pragmas -Wpointer-arith -Wmissing-declarations -Wredundant-decls -Wmultichar -Wno-unused-local-typedefs -Wunused")
    # 派生类中的虚函数声明中建议使用 override 关键字
    set(CXX_WARNING_FLAGS "-Wsuggest-override")
    # 禁止编译器在比较两个常量时发出警告
    set(CUDA_WARNING_FLAGS "-Wno-tautological-compare")
endif ()

# 设置 C++ 和 C 编译标志
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${C_WARNING_ERROR_FLAG} ${C_WARNING_FLAGS} ${CXX_WARNING_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${C_WARNING_ERROR_FLAG} ${C_WARNING_FLAGS}")
# 设置 CUDA 编译标志
if (MSVC)
    add_definitions(-DNOMINMAX)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_WARNING_ERROR_FLAG} ${CUDA_WARNING_FLAGS}")
else ()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_WARNING_ERROR_FLAG} ${C_WARNING_FLAGS} ${CXX_WARNING_FLAGS} ${CUDA_WARNING_FLAGS}")
endif ()

# 如果使用 GCC, 确保版本不低于 GCC 9.4, 否则给出错误并终止配置
# if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9.4)
#     message(FATAL_ERROR "Must use gcc>=9.4 to compile CV-CUDA, you're using ${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}")
# endif()

# 包含 CheckIPOSupported 模块
include(CheckIPOSupported)
# 检查当前编译器是否支持链接时间优化(LTO)
check_ipo_supported(RESULT LTO_SUPPORTED)


# 编译器是 GNU, 且版本大于等于 10.0, 开启 LTO
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"
   # Enable if gcc>=10. With 9.4 in some contexts we hit ICE with LTO:
   # internal compiler error: in add_symbol_to_partition_1, at lto/lto-partition.c:153
   AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10.0)
    set(LTO_ENABLED ON)
else()
    set(LTO_ENABLED OFF)
endif()

# 定义 ENABLE_SANITIZER 且编译器是 GCC, 开启 sanitizer 来检测代码问题
if(${PROJECT_NAME_UPPER}_ENABLE_SANITIZER AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
# -fsanitize=address：检测内存泄漏和越界访问。
# -fsanitize-address-use-after-scope：检测使用已经超出作用域的栈内存。
# -fsanitize=leak：检测内存泄漏。
# -fsanitize=undefined：检测未定义行为。
# -fno-sanitize-recover=all：禁用所有 sanitizer 的恢复机制。
# -static-liblsan 和 -static-libubsan：静态链接 liblsan 和 libubsan 库。
    set(COMPILER_SANITIZER_FLAGS
        -fsanitize=address
        -fsanitize-address-use-after-scope
        -fsanitize=leak
        -fsanitize=undefined
        -fno-sanitize-recover=all
        # not properly supported, see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=64234
        #-static-libasan
        -static-liblsan
        -static-libubsan)
    string(REPLACE ";" " " COMPILER_SANITIZER_FLAGS "${COMPILER_SANITIZER_FLAGS}" )

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMPILER_SANITIZER_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMPILER_SANITIZER_FLAGS}")
endif()

