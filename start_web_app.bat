@echo off
setlocal
cd /d "%~dp0"
title PDF Manga Translation System [Port 8000]

echo ========================================================
echo    PDF Manga Translation Service (Port 8000)
echo ========================================================
echo.

:: 0. Mode:  start_web_app.bat [cpu ^| gpu]   (no argument = auto: GPU only when one is detected)
::            start_cpu.bat / start_gpu.bat call this file with the argument.
set "MODE=%~1"
if "%MODE%"=="" set "MODE=auto"
if /i "%MODE%"=="cpu" (
    set "CPU_ONLY=1"
    set "LAMA_BACKEND=onnx"
    set "EXTRACT_DEVICE=cpu"
    set "MANGA_OCR_GPU_WORKER=0"
    echo Mode: CPU only - no GPU is used, the Qwen LLM is not started, translation uses Google.
) else if /i "%MODE%"=="gpu" (
    set "CPU_ONLY="
    set "LAMA_BACKEND=auto"
    set "EXTRACT_DEVICE=auto"
    set "MANGA_OCR_GPU_WORKER=1"
    echo Mode: GPU - LaMa, MangaOCR and the detector use the GPU when it has room, also in serial mode.
) else (
    echo Mode: auto - the GPU is used only when one is detected.
)
echo.

:: 1. Auto-release port 8000 if occupied by old instance
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo Releasing previously occupied Port 8000 PID %%p
    taskkill /f /pid %%p >nul 2>&1
)

:: 2. Auto-detect & start Turbovec Qwen 3.5 LLM engine (Port 18089 / 18088)
::    (skipped in CPU mode: the LLM runs on the GPU. An LLM that is already running is still used.)
if /i "%MODE%"=="cpu" goto after_llm
echo Checking LLM inference engine status (Port 18089 / 18088)...
netstat -aon | findstr ":18089" | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 goto llm_ready

netstat -aon | findstr ":18088" | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 goto llm_ready

echo Starting local Qwen 3.5 LLM engine in background...
if exist "%~dp0..\qwen_turbovec_rag\app\llm_launcher.py" (
    start /min "Turbovec LLM Engine" /d "%~dp0..\qwen_turbovec_rag" python app\llm_launcher.py --model 1
    echo Waiting for LLM engine to load model into memory...
    for /l %%i in (1,1,25) do (
        ping 127.0.0.1 -n 2 >nul
        netstat -aon | findstr ":18089" | findstr "LISTENING" >nul 2>&1
        if not errorlevel 1 goto llm_ready
        netstat -aon | findstr ":18088" | findstr "LISTENING" >nul 2>&1
        if not errorlevel 1 goto llm_ready
    )
)

:llm_ready
echo [OK] LLM inference engine ready!
:after_llm

:: 3. Launch browser
echo.
echo Launching browser at http://127.0.0.1:8000
start "" "http://127.0.0.1:8000"

:: 4. Start FastAPI backend
echo Starting FastAPI backend server
cd /d "%~dp0pdf_translate"
python main.py

pause
