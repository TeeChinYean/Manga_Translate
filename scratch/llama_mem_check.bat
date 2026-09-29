@echo off
rem Check llama-server RAM (Private vs Working set), launch flags and GPU memory.
rem Usage: double-click, or run  scratch\llama_mem_check.bat  from the project folder.
rem   scratch\llama_mem_check.bat restart   -> kill llama-server first (next translation relaunches it with new flags)
cd /d "%~dp0.."
if /i "%~1"=="restart" (
    taskkill /F /IM com.docker.llama-server.exe
    echo llama-server stopped. Start a translation in the web app, then run this bat again without "restart".
    pause
    exit /b 0
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0llama_mem_check.ps1"
pause
