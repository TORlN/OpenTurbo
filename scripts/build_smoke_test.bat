@echo off
setlocal

set "WORKSPACE_DIR=%~dp0.."
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSINSTALL="
set "VSDEVCMD="
set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe"
set "OUTPUT=%WORKSPACE_DIR%\build\encoder_smoke_test.exe"
set "ENCODER_CU=%WORKSPACE_DIR%\kernels\encoder.cu"
set "SMOKE_TEST_CU=%WORKSPACE_DIR%\kernels\encoder_smoke_test.cu"

if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" (
    set "VSDEVCMD=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
)

if not defined VSDEVCMD if exist "%VSWHERE%" (
    for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -version "[17.0,18.0)" -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        set "VSINSTALL=%%I"
    )
    if defined VSINSTALL (
        set "VSDEVCMD=%VSINSTALL%\Common7\Tools\VsDevCmd.bat"
    )
)

if not exist "%VSDEVCMD%" (
    echo CUDA build requires a supported Visual Studio 2022 C++ toolchain.
    echo Could not find VsDevCmd.bat for VS 2022 Build Tools.
    exit /b 1
)

if not exist "%NVCC%" (
    echo nvcc.exe not found at "%NVCC%"
    exit /b 1
)

if not exist "%WORKSPACE_DIR%\build" (
    mkdir "%WORKSPACE_DIR%\build"
)

call "%VSDEVCMD%" -no_logo -arch=amd64 -host_arch=amd64
if errorlevel 1 exit /b %errorlevel%

"%NVCC%" -m64 -std=c++20 -arch=sm_89 -lineinfo -o "%OUTPUT%" "%ENCODER_CU%" "%SMOKE_TEST_CU%"
exit /b %errorlevel%