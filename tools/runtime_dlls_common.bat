@echo off
rem Shared runtime DLL helper: link or copy ONNX Runtime / OpenVINO / CUDA /
rem TensorRT / OpenCV / Faiss dependencies into the build or install bin directory
rem for inferrt_model, inferrt_model_py, and sample executables.
rem Called by link_runtime_dlls.bat (link) and package_runtime_dlls.bat (copy).

setlocal EnableExtensions EnableDelayedExpansion

rem First argument: link = link into build\bin; copy = copy into install bin.
set "ACTION=%~1"
if /I not "%ACTION%"=="link" if /I not "%ACTION%"=="copy" (
    echo Usage: runtime_dlls_common.bat link^|copy
    exit /b 2
)

rem Default paths and mode; callers may override via environment variables.
if not defined SCRIPT_DIR set "SCRIPT_DIR=%~dp0"
if not defined PROJECT_ROOT for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"
if not defined BUILD_DIR set "BUILD_DIR=%PROJECT_ROOT%\build"
if not defined BUILD_CONFIG set "BUILD_CONFIG=Release"
if not defined LINK_MODE set "LINK_MODE=symlink"

rem Read dependency roots from cmake/Config*.cmake unless already set.
call :read_cmake_set "%PROJECT_ROOT%\cmake\ConfigONNXRuntime.cmake" "ONNXRUNTIME_ROOT" ONNXRUNTIME_ROOT
call :read_cmake_set "%PROJECT_ROOT%\cmake\ConfigOpenVINO.cmake" "INFERRT_OPENVINO_ROOT" OPENVINO_ROOT
call :read_cmake_set "%PROJECT_ROOT%\cmake\ConfigCUDA.cmake" "INFERRT_CUDA_ROOT" CUDA_ROOT
call :read_cmake_set "%PROJECT_ROOT%\cmake\ConfigTensorRT.cmake" "TRT_ROOT" TENSORRT_ROOT
call :read_cmake_set "%PROJECT_ROOT%\cmake\ConfigOpenCV.cmake" "OpenCV_HOME" OPENCV_HOME
call :read_cmake_set "%PROJECT_ROOT%\cmake\ConfigPython.cmake" "INFERRT_PYTHON_ROOT" PYTHON_ROOT
call :read_cmake_set "%PROJECT_ROOT%\cmake\ConfigFaiss.cmake" "Faiss_HOME" FAISS_HOME

rem Derive paths and apply fallback defaults.
if not defined OPENCV_BIN_DIR if defined OPENCV_HOME set "OPENCV_BIN_DIR=%OPENCV_HOME:/=\%\bin"
if not defined FAISS_BIN_DIR if defined FAISS_HOME set "FAISS_BIN_DIR=%FAISS_HOME:/=\%\bin"
if not defined TENSORRT_ROOT if defined TRT_ROOT set "TENSORRT_ROOT=%TRT_ROOT%"
if not defined CUDA_ROOT if defined CUDA_PATH set "CUDA_ROOT=%CUDA_PATH%"
if not defined PYTHON_ROOT if defined INFERRT_PYTHON_ROOT set "PYTHON_ROOT=%INFERRT_PYTHON_ROOT%"
if not defined ZLIBWAPI_DIR set "ZLIBWAPI_DIR=%PROJECT_ROOT%\ort1190_cuda118_dlls"
if not defined INSTALL_DIR call :read_cache_set "%BUILD_DIR%\CMakeCache.txt" "CMAKE_INSTALL_PREFIX" INSTALL_DIR
if not defined INSTALL_DIR set "INSTALL_DIR=%PROJECT_ROOT%\InferRT-0.0.1"

for %%I in ("%PROJECT_ROOT%") do set "PROJECT_ROOT=%%~fI"
for %%I in ("%BUILD_DIR%") do set "BUILD_DIR=%%~fI"
for %%I in ("%INSTALL_DIR%") do set "INSTALL_DIR=%%~fI"

rem link writes to build\bin; copy writes to install bin (override with INSTALL_BIN_DIR).
if /I "%ACTION%"=="link" (
    set "DEST_DIR=%BUILD_DIR%\bin"
) else (
    if defined INSTALL_BIN_DIR (
        set "DEST_DIR=%INSTALL_BIN_DIR%"
    ) else (
        set "DEST_DIR=%INSTALL_DIR%\bin"
    )
)

if not exist "%DEST_DIR%" mkdir "%DEST_DIR%"
if errorlevel 1 exit /b 1

set /a RUNTIME_COUNT=0
set "FAILED="

rem In copy mode, collect InferRT artifacts first, then third-party runtime DLLs.
if /I "%ACTION%"=="copy" (
    call :handle_dir "%BUILD_DIR%\bin" "inferrt*.dll"
    call :handle_dir "%BUILD_DIR%\bin" "inferrt_model_py*.pyd"
)

if defined ONNXRUNTIME_ROOT call :handle_dir "%ONNXRUNTIME_ROOT%\lib" "*.dll"
if defined TENSORRT_ROOT call :handle_dir "%TENSORRT_ROOT%\bin" "*.dll"
if defined OPENCV_BIN_DIR call :handle_dir "%OPENCV_BIN_DIR%" "*.dll"
if defined FAISS_BIN_DIR call :handle_dir "%FAISS_BIN_DIR%" "*.dll"

rem OpenVINO runtime, cache.json, and TBB dependencies.
if defined OPENVINO_ROOT call :handle_dir "%OPENVINO_ROOT%\runtime\bin\intel64\%BUILD_CONFIG%" "*.dll"
if defined OPENVINO_ROOT call :handle_file "%OPENVINO_ROOT%\runtime\bin\intel64\%BUILD_CONFIG%\cache.json"
if defined OPENVINO_ROOT call :handle_dir "%OPENVINO_ROOT%\runtime\3rdparty\tbb\bin" "tbb12.dll"
if defined OPENVINO_ROOT call :handle_dir "%OPENVINO_ROOT%\runtime\3rdparty\tbb\bin" "tbbbind_2_5.dll"
if defined OPENVINO_ROOT call :handle_dir "%OPENVINO_ROOT%\runtime\3rdparty\tbb\bin" "tbbmalloc.dll"
if defined OPENVINO_ROOT call :handle_dir "%OPENVINO_ROOT%\runtime\3rdparty\tbb\bin" "tbbmalloc_proxy.dll"

rem CUDA / cuDNN and zlibwapi.dll for ORT GPU provider.
if defined CUDA_ROOT call :handle_dir "%CUDA_ROOT%\bin" "cudart64_*.dll"
if defined CUDA_ROOT call :handle_dir "%CUDA_ROOT%\bin" "cublas64_*.dll"
if defined CUDA_ROOT call :handle_dir "%CUDA_ROOT%\bin" "cublasLt64_*.dll"
if defined CUDA_ROOT call :handle_dir "%CUDA_ROOT%\bin" "cufft64_*.dll"
if defined CUDA_ROOT call :handle_dir "%CUDA_ROOT%\bin" "cudnn*.dll"
if defined CUDA_ROOT call :handle_dir "%CUDA_ROOT%\bin" "zlibwapi.dll"

if defined ZLIBWAPI_DIR call :handle_dir "%ZLIBWAPI_DIR%" "zlibwapi.dll"
if defined PYTHON_ROOT call :handle_file "%PYTHON_ROOT%\Lib\site-packages\torch\lib\zlibwapi.dll"

echo.
if defined FAILED (
    echo Runtime %ACTION% finished with errors. Destination: %DEST_DIR%
    exit /b 1
)
echo Runtime %ACTION% finished. Files processed: %RUNTIME_COUNT%
echo Destination: %DEST_DIR%
exit /b 0

rem Iterate files matching PATTERN under SRC_DIR and pass each to :handle_file.
:handle_dir
set "SRC_DIR=%~1"
set "PATTERN=%~2"
if "%SRC_DIR%"=="" exit /b 0
if not exist "%SRC_DIR%\" (
    echo [skip] missing dir: %SRC_DIR%
    exit /b 0
)
for %%F in ("%SRC_DIR%\%PATTERN%") do (
    if exist "%%~fF" call :handle_file "%%~fF"
)
exit /b 0

rem Link or copy a single source file into DEST_DIR; process each name once.
:handle_file
set "SRC=%~1"
if "%SRC%"=="" exit /b 0
if not exist "%SRC%" exit /b 0
for %%F in ("%SRC%") do set "NAME=%%~nxF"
rem Skip Debug variants such as inferrt_*d.dll, opencv_*d.dll, and faissd.dll in Release builds.
if /I "!BUILD_CONFIG!"=="Release" if /I "!NAME:~0,8!"=="inferrt_" if /I "!NAME:~-5!"=="d.dll" exit /b 0
if /I "!BUILD_CONFIG!"=="Release" if /I "!NAME:~0,7!"=="opencv_" if /I "!NAME:~-5!"=="d.dll" exit /b 0
if /I "!BUILD_CONFIG!"=="Release" if /I "!NAME!"=="faissd.dll" exit /b 0
if defined SEEN_!NAME! exit /b 0
set "DST=!DEST_DIR!\!NAME!"

if /I "!ACTION!"=="copy" (
    copy /Y "!SRC!" "!DST!" >nul
    if errorlevel 1 (
        echo [error] copy failed: !SRC!
        set "FAILED=1"
    ) else (
        echo [copy] !NAME!
        set "SEEN_!NAME!=1"
        set /a RUNTIME_COUNT+=1
    )
    exit /b 0
)

rem In link mode, remove an existing target, then link with automatic fallback.
if exist "!DST!" del /F /Q "!DST!" >nul 2>nul
if exist "!DST!" (
    echo [error] cannot replace: !DST!
    set "FAILED=1"
    exit /b 0
)

if /I "!LINK_MODE!"=="copy" (
    copy /Y "!SRC!" "!DST!" >nul
    if errorlevel 1 (
        echo [error] copy failed: !SRC!
        set "FAILED=1"
    ) else (
        echo [copy] !NAME!
        set "SEEN_!NAME!=1"
        set /a RUNTIME_COUNT+=1
    )
    exit /b 0
)

if /I "!LINK_MODE!"=="hardlink" (
    mklink /H "!DST!" "!SRC!" >nul 2>nul
    if errorlevel 1 (
        echo [warn] hardlink failed, copying: !NAME!
        copy /Y "!SRC!" "!DST!" >nul
        if errorlevel 1 (
            echo [error] copy fallback failed: !SRC!
            set "FAILED=1"
        ) else (
            echo [copy] !NAME!
            set "SEEN_!NAME!=1"
            set /a RUNTIME_COUNT+=1
        )
    ) else (
        echo [hardlink] !NAME!
        set "SEEN_!NAME!=1"
        set /a RUNTIME_COUNT+=1
    )
    exit /b 0
)

rem Default: symlink, then hardlink, then copy.
mklink "!DST!" "!SRC!" >nul 2>nul
if errorlevel 1 (
    echo [warn] symlink failed, trying hardlink: !NAME!
    mklink /H "!DST!" "!SRC!" >nul 2>nul
    if errorlevel 1 (
        echo [warn] hardlink failed, copying: !NAME!
        copy /Y "!SRC!" "!DST!" >nul
        if errorlevel 1 (
            echo [error] copy fallback failed: !SRC!
            set "FAILED=1"
        ) else (
            echo [copy] !NAME!
            set "SEEN_!NAME!=1"
            set /a RUNTIME_COUNT+=1
        )
    ) else (
        echo [hardlink] !NAME!
        set "SEEN_!NAME!=1"
        set /a RUNTIME_COUNT+=1
    )
) else (
    echo [link] !NAME!
    set "SEEN_!NAME!=1"
    set /a RUNTIME_COUNT+=1
)
exit /b 0

rem Parse set(KEY "value") or set(KEY value) defaults from Config*.cmake.
:read_cmake_set
set "CFG_FILE=%~1"
set "CFG_KEY=%~2"
set "CFG_OUT=%~3"
if defined %CFG_OUT% exit /b 0
if not exist "%CFG_FILE%" exit /b 0
set "CFG_FOUND_LINE="
for /f "usebackq delims=" %%L in ("%CFG_FILE%") do (
    set "CFG_LINE=%%L"
    if not defined CFG_FOUND_LINE if /I "!CFG_LINE:~0,4!"=="set(" (
        set "CFG_REST=!CFG_LINE:~4!"
        for /f "tokens=1,* delims= " %%A in ("!CFG_REST!") do (
            if /I "%%A"=="%CFG_KEY%" (
                set "CFG_FOUND_LINE=!CFG_LINE!"
            )
        )
    )
)
if not defined CFG_FOUND_LINE exit /b 0
for /f tokens^=2^ delims^=^" %%V in ("!CFG_FOUND_LINE!") do set "%CFG_OUT%=%%V"
if defined %CFG_OUT% exit /b 0
set "CFG_VALUE_TEXT=!CFG_FOUND_LINE:*%CFG_KEY%=!"
for /f "tokens=1 delims= )" %%V in ("!CFG_VALUE_TEXT!") do set "%CFG_OUT%=%%V"
exit /b 0

rem Read KEY=VALUE entries from CMakeCache.txt.
:read_cache_set
set "CACHE_FILE=%~1"
set "CACHE_KEY=%~2"
set "CACHE_OUT=%~3"
if defined %CACHE_OUT% exit /b 0
if not exist "%CACHE_FILE%" exit /b 0
for /f "usebackq tokens=1,* delims==" %%A in ("%CACHE_FILE%") do (
    for /f "tokens=1 delims=:" %%K in ("%%A") do (
        if /I "%%K"=="%CACHE_KEY%" (
            set "%CACHE_OUT%=%%B"
            exit /b 0
        )
    )
)
exit /b 0
