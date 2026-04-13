@echo off
setlocal

set "WORKSPACE_DIR=%~dp0.."
set "VSDEVCMD=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat"
set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe"
set "OUTPUT=%WORKSPACE_DIR%\build\encoder_smoke_test.exe"
set "ENCODER_CU=%WORKSPACE_DIR%\kernels\encoder.cu"
set "SMOKE_TEST_CU=%WORKSPACE_DIR%\kernels\encoder_smoke_test.cu"

if not exist "%VSDEVCMD%" (
    echo VsDevCmd.bat not found at "%VSDEVCMD%"
    exit /b 1
)

if not exist "%NVCC%" (
    echo nvcc.exe not found at "%NVCC%"
    exit /b 1
)

if not exist "%WORKSPACE_DIR%\build" (
    mkdir "%WORKSPACE_DIR%\build"
)

call "%VSDEVCMD%" -no_logo
if errorlevel 1 exit /b %errorlevel%

"%NVCC%" -allow-unsupported-compiler -std=c++20 -arch=sm_89 -lineinfo -o "%OUTPUT%" "%ENCODER_CU%" "%SMOKE_TEST_CU%"
exit /b %errorlevel%