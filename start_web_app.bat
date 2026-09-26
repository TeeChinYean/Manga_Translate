@echo off
chcp 65001 >nul
title PDF 漫画翻译系统 [Port 8000]

echo ========================================================
echo    PDF 漫画翻译与排版保真引擎 (Port 8000)
echo ========================================================
echo.

cd /d "%~dp0pdf_translate"

echo 正在启动服务 (http://127.0.0.1:8000)...
echo 提示: 浏览器将在 2 秒后自动打开。
echo.

start "" "http://127.0.0.1:8000"

python main.py

pause
