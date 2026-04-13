@echo off
setlocal

set "WORKSPACE_DIR=%~dp0.."
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSINSTALL="
set "VSDEVCMD="
set "NVCC="
set "PYTHON=%WORKSPACE_DIR%\.venv\Scripts\python.exe"

if defined CUDA_PATH if exist "%CUDA_PATH%\bin\nvcc.exe" (
    set "NVCC=%CUDA_PATH%\bin\nvcc.exe"
)

if not defined NVCC if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" (
    set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe"
)

if not defined NVCC (
    for /f "delims=" %%I in ('where nvcc 2^>nul') do (
        set "NVCC=%%I"
        goto :found_nvcc
    )
)

:found_nvcc

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
    echo CUDA extension build requires a supported Visual Studio 2022 C++ toolchain.
    echo Could not find VsDevCmd.bat for VS 2022 Build Tools.
    exit /b 1
)

if not exist "%NVCC%" (
    echo nvcc.exe not found at "%NVCC%"
    exit /b 1
)

if not exist "%PYTHON%" (
    set "PYTHON=python"
)

call "%VSDEVCMD%" -no_logo -arch=amd64 -host_arch=amd64
if errorlevel 1 exit /b %errorlevel%

set "CUDACXX=%NVCC%"
set "CMAKE_GENERATOR=Ninja"
set "CMAKE_ARGS=-DCMAKE_CUDA_ARCHITECTURES=89"

pushd "%WORKSPACE_DIR%"
"%PYTHON%" -m pip install -e .[bindings]
set "RESULT=%ERRORLEVEL%"
popd

exit /b %RESULT%
