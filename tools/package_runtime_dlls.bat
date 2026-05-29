@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"

rem ===================== User settings =====================
rem Runtime dependency roots are read from cmake/Config*.cmake:
rem   ConfigONNXRuntime.cmake -> ONNXRUNTIME_ROOT
rem   ConfigOpenVINO.cmake    -> INFERRT_OPENVINO_ROOT
rem   ConfigCUDA.cmake        -> INFERRT_CUDA_ROOT
rem   ConfigTensorRT.cmake    -> TRT_ROOT
rem   ConfigOpenCV.cmake      -> OpenCV_HOME
rem   ConfigPython.cmake      -> INFERRT_PYTHON_ROOT
rem   ConfigFaiss.cmake       -> Faiss_HOME
rem INSTALL_DIR defaults to CMAKE_INSTALL_PREFIX from build/CMakeCache.txt.
set "BUILD_DIR=%PROJECT_ROOT%\build"
set "BUILD_CONFIG=Release"
rem =========================================================

call "%SCRIPT_DIR%runtime_dlls_common.bat" copy
exit /b %ERRORLEVEL%
