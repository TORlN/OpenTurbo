@echo off
setlocal

set "WORKSPACE_DIR=%~dp0.."
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSINSTALL="
set "VSDEVCMD="
set "NVCC="
set "OUTPUT=%WORKSPACE_DIR%\build\c_api_smoke_test.exe"
set "ENCODER_CU=%WORKSPACE_DIR%\kernels\encoder.cu"
set "SCAN_CU=%WORKSPACE_DIR%\kernels\scan.cu"
set "C_API_CU=%WORKSPACE_DIR%\kernels\openturbo_c_api.cu"
set "GGML_ADAPTER_CPP=%WORKSPACE_DIR%\kernels\openturbo_ggml_adapter.cpp"
set "LLAMA_BRIDGE_CPP=%WORKSPACE_DIR%\kernels\openturbo_llama_bridge.cpp"
set "LLAMA_SHIM_CPP=%WORKSPACE_DIR%\kernels\openturbo_llama_kv_shim.cpp"
set "SMOKE_TEST_CPP=%WORKSPACE_DIR%\kernels\c_api_smoke_test.cpp"

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

"%NVCC%" -m64 -std=c++20 -arch=sm_89 -lineinfo -DOPENTURBO_CAPI_EXPORTS -I"%WORKSPACE_DIR%\include" -o "%OUTPUT%" "%ENCODER_CU%" "%SCAN_CU%" "%C_API_CU%" "%GGML_ADAPTER_CPP%" "%LLAMA_BRIDGE_CPP%" "%LLAMA_SHIM_CPP%" "%SMOKE_TEST_CPP%"
exit /b %errorlevel%
