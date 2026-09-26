@echo off
setlocal
cd /d "%~dp0"
title PDF Manga Translation System [Port 8000]

echo ========================================================
echo    PDF Manga Translation Service (Port 8000)
echo ========================================================
echo.

:: 1. Auto-release port 8000 if occupied by old instance
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo Releasing previously occupied Port 8000 PID %%p
    taskkill /f /pid %%p >nul 2>&1
)

:: 2. Auto-detect Turbovec Qwen 3.5 LLM engine (Port 18089 / 18088)
netstat -aon | findstr ":18089" | findstr "LISTENING" >nul 2>&1
if errorlevel 1 (
    echo Checking Turbovec gateway...
    netstat -aon | findstr ":18088" | findstr "LISTENING" >nul 2>&1
    if errorlevel 1 (
        echo Starting local Qwen 3.5 LLM engine
        if exist "%~dp0..\qwen_turbovec_rag\app\llm_launcher.py" (
            start /min "Turbovec LLM Engine" python "%~dp0..\qwen_turbovec_rag\app\llm_launcher.py" --model 1
            timeout /t 3 /nobreak >nul
        )
    )
)

:: 3. Launch browser
echo.
echo Launching browser at http://127.0.0.1:8000
start "" "http://127.0.0.1:8000"

:: 4. Start FastAPI backend
echo Starting FastAPI backend server
cd /d "%~dp0pdf_translate"
python main.py

pause
