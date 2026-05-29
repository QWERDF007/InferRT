@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"

rem ===================== 用户配置 =====================
rem 运行时依赖路径从 cmake/Config*.cmake 读取：
rem   ConfigONNXRuntime.cmake -> ONNXRUNTIME_ROOT
rem   ConfigOpenVINO.cmake    -> INFERRT_OPENVINO_ROOT
rem   ConfigCUDA.cmake        -> INFERRT_CUDA_ROOT
rem   ConfigTensorRT.cmake    -> TRT_ROOT
rem   ConfigOpenCV.cmake      -> OpenCV_HOME
rem   ConfigPython.cmake      -> INFERRT_PYTHON_ROOT
rem INSTALL_DIR 默认取 build/CMakeCache.txt 中的 CMAKE_INSTALL_PREFIX。
set "BUILD_DIR=%PROJECT_ROOT%\build"
set "BUILD_CONFIG=Release"
rem =====================================================

call "%SCRIPT_DIR%runtime_dlls_common.bat" copy
exit /b %ERRORLEVEL%
