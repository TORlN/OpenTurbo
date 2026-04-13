@echo off
setlocal

set "WORKSPACE_DIR=%~dp0.."
set "PYTHON=%WORKSPACE_DIR%\.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo Python virtual environment not found at "%PYTHON%".
    echo Create or activate the OpenTurbo .venv first.
    exit /b 1
)

echo For the full one-command bootstrap, build, model-download, and probe-run flow use:
echo   %PYTHON% "%WORKSPACE_DIR%\scripts\run_llama_cpp_k_cache_probe.py"
echo.

if "%~1"=="" (
    "%PYTHON%" "%WORKSPACE_DIR%\scripts\scaffold_llama_cpp_integration.py" --bootstrap --probe-k-cache --force
) else (
    "%PYTHON%" "%WORKSPACE_DIR%\scripts\scaffold_llama_cpp_integration.py" --llama-root "%~1" --bootstrap --probe-k-cache --force
)

exit /b %errorlevel%