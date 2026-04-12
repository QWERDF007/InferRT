# 测试通用库目标名称
set(TARGET_NAME test_common)

# ============================================================================
# 依赖库配置
# ============================================================================

# 查找 OpenSSL 库
# 提示：如果找不到 OpenSSL，请设置环境变量 OPENSSL_ROOT_DIR
# 例如：set OPENSSL_ROOT_DIR=C:\Program Files\OpenSSL-Win64
# 下载地址：https://slproweb.com/products/Win32OpenSSL.html
# 或使用 vcpkg 安装: vcpkg install openssl:x64-windows-static
set(OPENSSL_ROOT_DIR "E:/Softwares/OpenSSL-Win64")
set(OPENSSL_USE_STATIC_LIBS TRUE)
find_package(OpenSSL COMPONENTS Crypto REQUIRED)

# 查找 zlib 库（静态链接 OpenSSL 时可能需要）
# 如果找不到 zlib，可以尝试：
# 1. 使用 vcpkg 安装: vcpkg install zlib:x64-windows-static
# 2. 或者将 REQUIRED 改为可选
set(ZLIB_USE_STATIC_LIBS TRUE)
find_package(ZLIB)

# ============================================================================
# 源文件收集
# ============================================================================

# 递归获取所有 C++ 源文件的相对路径
file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

# 递归获取所有头文件的相对路径
file(GLOB_RECURSE HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h *.hpp)


# ============================================================================
# 创建静态库
# ============================================================================

# 创建测试通用静态库，包含所有源文件和头文件
add_library(${TARGET_NAME} STATIC
    ${SOURCES}
    ${HEADERS}
)

# ============================================================================
# 链接依赖库
# ============================================================================

# 链接 OpenSSL 加密组件（只需要加密功能，不需要 SSL/TLS）
target_link_libraries(${TARGET_NAME}
    PRIVATE
        OpenSSL::Crypto
)

# 如果找到了 ZLIB，则链接它（OpenSSL 静态链接时可能需要）
if(ZLIB_FOUND)
    target_link_libraries(${TARGET_NAME} PRIVATE ZLIB::ZLIB)
    message(STATUS "ZLIB found and linked")
else()
    message(WARNING "ZLIB not found, linking without it. If you encounter linking errors, please install zlib.")
endif()

# ============================================================================
# 配置头文件包含路径
# ============================================================================

# 创建接口库用于头文件包含
# 其他目标链接此接口库后，可以直接 #include 对应头文件，无需手动设置 include 目录
set(TARGET_HEADER "${TARGET_NAME}_header")
add_library(${TARGET_HEADER} INTERFACE)
target_include_directories(${TARGET_HEADER}
    INTERFACE 
        ${CMAKE_CURRENT_SOURCE_DIR}/include 
        ${CMAKE_CURRENT_BINARY_DIR}/include
)

# 将接口库链接到主库，使头文件路径对外可见
target_link_libraries(${TARGET_NAME} PUBLIC ${TARGET_HEADER})