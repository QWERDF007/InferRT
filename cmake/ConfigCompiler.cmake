
# RelWithDebInfo 优化选项由目标 helper 注入，避免污染第三方目标。


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
    set(INFERRT_CXX_COMPILE_OPTIONS /EHa /utf-8 /bigobj /W4 /wd4251)
    set(INFERRT_C_COMPILE_OPTIONS /W4)
    set(INFERRT_CUDA_COMPILE_OPTIONS)
    # set(CXX_WARNING_FLAGS "/permissive-")
else ()
    set(INFERRT_CXX_COMPILE_OPTIONS ${C_WARNING_ERROR_FLAG} -Wall -Wno-unknown-pragmas -Wpointer-arith -Wmissing-declarations -Wredundant-decls -Wmultichar -Wno-unused-local-typedefs -Wunused -Wsuggest-override)
    set(INFERRT_C_COMPILE_OPTIONS ${C_WARNING_ERROR_FLAG} -Wall -Wno-unknown-pragmas -Wpointer-arith -Wmissing-declarations -Wredundant-decls -Wmultichar -Wno-unused-local-typedefs -Wunused)
    set(INFERRT_CUDA_COMPILE_OPTIONS ${CUDA_WARNING_ERROR_FLAG} -Wall -Wno-unknown-pragmas -Wpointer-arith -Wmissing-declarations -Wredundant-decls -Wmultichar -Wno-unused-local-typedefs -Wunused -Wsuggest-override -Wno-tautological-compare)
    # 派生类中的虚函数声明中建议使用 override 关键字
    set(CXX_WARNING_FLAGS "-Wsuggest-override")
    # 禁止编译器在比较两个常量时发出警告
    set(CUDA_WARNING_FLAGS "-Wno-tautological-compare")
endif ()

function(inferrt_apply_compile_options target)
    target_compile_features(${target} PRIVATE cxx_std_20)
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    foreach(option IN LISTS INFERRT_CXX_COMPILE_OPTIONS)
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${option}>")
    endforeach()
    foreach(option IN LISTS INFERRT_C_COMPILE_OPTIONS)
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:C>:${option}>")
    endforeach()
    foreach(option IN LISTS INFERRT_SANITIZER_COMPILE_OPTIONS)
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:${option}>")
    endforeach()
    if(INFERRT_SANITIZER_LINK_OPTIONS)
        target_link_options(${target} PRIVATE ${INFERRT_SANITIZER_LINK_OPTIONS})
        target_compile_definitions(${target} PRIVATE ENABLE_SANITIZER=1)
    endif()
    if(NOT MSVC)
        target_compile_options(${target} PRIVATE
            "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RelWithDebInfo>>:-O3>"
            "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RelWithDebInfo>>:-ggdb>"
        )
    endif()
    if(CMAKE_CUDA_COMPILER)
        set_target_properties(${target} PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
        foreach(option IN LISTS INFERRT_CUDA_COMPILE_OPTIONS INFERRT_CUDA_TOOLKIT_OPTIONS)
            target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${option}>")
        endforeach()
    endif()
    if(${PROJECT_NAME_UPPER}_ENABLE_CUDA)
        target_compile_definitions(${target} PRIVATE INFERRT_ENABLE_CUDA=1)
    endif()
    if(${PROJECT_NAME_UPPER}_BUILD_ONNX)
        target_compile_definitions(${target} PRIVATE INFERRT_BUILD_ONNX=1)
    else()
        target_compile_definitions(${target} PRIVATE INFERRT_BUILD_ONNX=0)
    endif()
    if(${PROJECT_NAME_UPPER}_BUILD_OPENVINO)
        target_compile_definitions(${target} PRIVATE INFERRT_BUILD_OPENVINO=1)
    else()
        target_compile_definitions(${target} PRIVATE INFERRT_BUILD_OPENVINO=0)
    endif()
    if(MSVC)
        target_compile_definitions(${target} PRIVATE NOMINMAX)
    endif()
endfunction()

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
    set(INFERRT_SANITIZER_COMPILE_OPTIONS
        -fsanitize=address
        -fsanitize-address-use-after-scope
        -fsanitize=undefined
        -fno-sanitize-recover=all
        -fno-omit-frame-pointer)
    set(INFERRT_SANITIZER_LINK_OPTIONS
        -fsanitize=address
        -fsanitize=undefined
        -fno-sanitize-recover=all)
endif()

