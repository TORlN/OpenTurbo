@echo off
setlocal

set "WORKSPACE_DIR=%~dp0.."
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSINSTALL="
set "VSDEVCMD="
set "NVCC="
set "CUDA_TOOLKIT_ROOT="
set "PYTHON=%WORKSPACE_DIR%\.venv\Scripts\python.exe"
set "HAS_CL="
set "HOST_CL="
set "NINJA="

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

for %%I in ("%NVCC%") do set "CUDA_BIN_DIR=%%~dpI"
for %%I in ("%CUDA_BIN_DIR%..") do set "CUDA_TOOLKIT_ROOT=%%~fI"

if not exist "%PYTHON%" (
    set "PYTHON=python"
)

where cl >nul 2>nul
if not errorlevel 1 (
    set "HAS_CL=1"
)

if not defined HAS_CL (
    call "%VSDEVCMD%" -no_logo -arch=amd64 -host_arch=amd64
    if errorlevel 1 exit /b %errorlevel%
)

for /f "delims=" %%I in ('where cl 2^>nul') do (
    set "HOST_CL=%%I"
    goto :found_cl
)

:found_cl

for /f "delims=" %%I in ('where ninja 2^>nul') do (
    set "NINJA=%%I"
    goto :found_ninja
)

if not defined NINJA if defined VSINSTALL if exist "%VSINSTALL%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" (
    set "NINJA=%VSINSTALL%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
)

if not defined NINJA if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" (
    set "NINJA=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
)

:found_ninja

set "CUDACXX=%NVCC%"
set "CUDAHOSTCXX=%HOST_CL%"
set "CMAKE_GENERATOR_PLATFORM="
set "CMAKE_GENERATOR_TOOLSET="
set "CMAKE_MAKE_PROGRAM="

if defined HOST_CL if defined NINJA (
    set "CMAKE_GENERATOR=Ninja"
    set "CMAKE_MAKE_PROGRAM=%NINJA%"
) else (
    set "CMAKE_GENERATOR=Visual Studio 17 2022"
    set "CMAKE_GENERATOR_PLATFORM=x64"
    if defined CUDA_TOOLKIT_ROOT (
        set "CMAKE_GENERATOR_TOOLSET=cuda=%CUDA_TOOLKIT_ROOT%"
    )
)

set "CMAKE_ARGS=-DCMAKE_CUDA_ARCHITECTURES=89 -DCUDAToolkit_ROOT=%CUDA_TOOLKIT_ROOT%"

echo Using CMake generator: %CMAKE_GENERATOR%
if defined CMAKE_MAKE_PROGRAM echo Using make program: %CMAKE_MAKE_PROGRAM%
if defined HOST_CL echo Using host compiler: %HOST_CL%
if defined CUDA_TOOLKIT_ROOT echo Using CUDA toolkit root: %CUDA_TOOLKIT_ROOT%

pushd "%WORKSPACE_DIR%"
"%PYTHON%" -m pip install -e .[bindings]
set "RESULT=%ERRORLEVEL%"
popd

exit /b %RESULT%
