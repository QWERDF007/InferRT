set(CMAKE_FOLDER 3rdparty)

# GoogleTest -----------------------------
set(INSTALL_GTEST off)
set(BUILD_GMOCK on)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
if(INFERRT_BUILD_TESTS)
    add_subdirectory(googletest)
endif()

# vtest -----------------------------
# 父项目使用了 EXCLUDE_FROM_ALL，会导致该子目录中的所有目标不会被默认构建，
# 除非它们被其他目标依赖。
# set(OPENSSL_ROOT_DIR "E:/Software/OpenSSL-Win64")
# 
# CMake 的 FindOpenSSL.cmake 模块会在 Windows 注册表中查找 OpenSSL 的安装路径，具体是在：
# HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\OpenSSL (64-bit)_is1
# 这是 Inno Setup 安装程序（OpenSSL Windows 安装包使用的安装器）在注册表中记录的安装路径。
add_subdirectory(vtest)