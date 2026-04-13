@echo off
setlocal

set "WORKSPACE_DIR=%~dp0.."
set "PYTHON=%WORKSPACE_DIR%\.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo Python virtual environment not found at "%PYTHON%".
    echo Create or activate the OpenTurbo .venv first.
    exit /b 1
)

if "%~1"=="" (
    "%PYTHON%" "%WORKSPACE_DIR%\scripts\scaffold_llama_cpp_integration.py" --bootstrap --force
) else (
    "%PYTHON%" "%WORKSPACE_DIR%\scripts\scaffold_llama_cpp_integration.py" --llama-root "%~1" --bootstrap --force
)

exit /b %errorlevel%