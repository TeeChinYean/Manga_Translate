@echo off
setlocal
cd /d "%~dp0"
title PDF Manga Translation System [Port 8000]

echo ========================================================
echo    PDF Manga Translation Service (Port 8000)
echo ========================================================
echo.
echo Launching browser at http://127.0.0.1:8000 ...
start "" "http://127.0.0.1:8000"

echo Starting FastAPI backend server...
cd /d "%~dp0pdf_translate"
python main.py

pause
