@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "MODE=%~1"
if "%MODE%"=="" set "MODE=auto"
set "REQUESTED_MODE=%MODE%"
set "APP_PORT=%APP_PORT%"
if "%APP_PORT%"=="" set "APP_PORT=8501"
set "APP_URL=http://localhost:%APP_PORT%"
set "COMPOSE_DIR=docker"

if /I "%MODE%"=="--help" (
    call :usage_text
    exit /b 0
)
if /I "%MODE%"=="-h" (
    call :usage_text
    exit /b 0
)

set "MODE_VALID="
if /I "%MODE%"=="auto" set "MODE_VALID=1"
if /I "%MODE%"=="native" set "MODE_VALID=1"
if /I "%MODE%"=="cpu" set "MODE_VALID=1"
if /I "%MODE%"=="nvidia" set "MODE_VALID=1"
if /I "%MODE%"=="amd" set "MODE_VALID=1"
if not defined MODE_VALID goto :usage

where docker >nul 2>nul
if errorlevel 1 (
    echo Error: Docker is not installed. Install Docker Desktop first.
    exit /b 1
)

docker info >nul 2>nul
if errorlevel 1 (
    echo Error: Docker is not running. Start Docker Desktop and run this file again.
    exit /b 1
)

docker compose version >nul 2>nul
if errorlevel 1 (
    echo Error: Docker Compose is not available. Update Docker Desktop.
    exit /b 1
)

for /f "delims=" %%G in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "try { (Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join ', ' } catch { '' }" 2^>nul') do set "GPU_NAMES=%%G"
if not "%GPU_NAMES%"=="" echo Detected GPU^(s^): %GPU_NAMES%

if /I "%MODE%"=="auto" (
    rem Default to an isolated Ollama container. Use NVIDIA only when clearly detected.
    set "MODE=cpu"
    echo(!GPU_NAMES! | findstr /I "NVIDIA RTX GeForce Quadro Tesla" >nul
    if not errorlevel 1 set "MODE=nvidia"
)

set "FILES=-f %COMPOSE_DIR%\compose.yaml"

if /I "%MODE%"=="native" (
    set "OLLAMA_BASE_URL=http://host.docker.internal:11434"
    call :ensure_native_ollama || exit /b 1
    echo Mode: native Ollama on host
) else if /I "%MODE%"=="cpu" (
    set "FILES=!FILES! -f %COMPOSE_DIR%\compose.ollama.yaml"
    echo Mode: Ollama CPU container
) else if /I "%MODE%"=="nvidia" (
    set "FILES=!FILES! -f %COMPOSE_DIR%\compose.ollama.yaml -f %COMPOSE_DIR%\compose.nvidia.yaml"
    echo Mode: Ollama NVIDIA GPU container
    echo This requires Docker Desktop with WSL2, current NVIDIA drivers, and GPU support enabled.
) else if /I "%MODE%"=="amd" (
    echo AMD GPU containers are Linux-only in this stack. Using the Ollama CPU container on Windows.
    set "MODE=cpu"
    set "FILES=!FILES! -f %COMPOSE_DIR%\compose.ollama.yaml"
) else (
    goto :usage
)

echo Building and starting containers...
docker compose %FILES% up --build -d
if errorlevel 1 (
    if /I "%REQUESTED_MODE%"=="auto" (
        if /I "%MODE%"=="nvidia" (
            echo NVIDIA container mode failed. Retrying with the Ollama CPU container.
            set "MODE=cpu"
            set "FILES=-f %COMPOSE_DIR%\compose.yaml -f %COMPOSE_DIR%\compose.ollama.yaml"
            docker compose !FILES! up --build -d
            if errorlevel 1 exit /b 1
        ) else (
            exit /b 1
        )
    ) else (
        exit /b 1
    )
)

if /I "%MODE%"=="nvidia" (
    docker compose %FILES% exec -T ollama nvidia-smi >nul 2>nul
    if errorlevel 1 (
        echo Warning: the Ollama container started, but NVIDIA GPU access was not visible inside it.
        echo Check Docker Desktop WSL2 GPU support, update NVIDIA drivers, then try: run.bat nvidia
        echo Diagnostic command: docker compose %FILES% exec -T ollama nvidia-smi
    ) else (
        echo NVIDIA GPU is visible inside the Ollama container.
    )
)

start "" "%APP_URL%"
echo App: %APP_URL%
echo Use "docker compose %FILES% logs -f" to view logs.
exit /b 0

:ensure_native_ollama
call :ollama_running
if not errorlevel 1 exit /b 0

call :find_ollama
if errorlevel 1 (
    echo Error: Ollama is not installed or not in PATH.
    echo Install it from https://ollama.com/download, open Ollama once, then run this file again.
    exit /b 1
)

echo Starting native Ollama on the host...
if not exist ".docker" mkdir ".docker" >nul 2>nul
start "Ollama" /min "%OLLAMA_CMD%" serve
timeout /t 5 /nobreak >nul

call :ollama_running
if errorlevel 1 (
    echo Error: Ollama did not answer at http://127.0.0.1:11434.
    exit /b 1
)
exit /b 0

:ollama_running
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri 'http://127.0.0.1:11434/api/tags' | Out-Null; exit 0 } catch { exit 1 }"
exit /b %ERRORLEVEL%

:find_ollama
where ollama >nul 2>nul
if not errorlevel 1 (
    set "OLLAMA_CMD=ollama"
    exit /b 0
)
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
    set "OLLAMA_CMD=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
    exit /b 0
)
exit /b 1

:usage
call :usage_text
exit /b 1

:usage_text
echo Usage:
echo   run.bat [auto^|native^|cpu^|nvidia^|amd]
echo.
echo Modes:
echo   auto     use the isolated Ollama container stack; NVIDIA if supported, otherwise CPU.
echo   native   run only the app in Docker and use Ollama from Windows.
echo   cpu      run app + Ollama CPU containers.
echo   nvidia   run app + Ollama container with NVIDIA GPU support.
echo   amd      use CPU container on Windows; AMD container mode is Linux-only here.
exit /b 0
