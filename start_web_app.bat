@echo off
chcp 65001 >nul
title PDF 漫画翻译系统 · 极速启动器

echo ========================================================
echo    ⚡ PDF 翻译与排版保真引擎 · Web 服务启动器
echo ========================================================
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_web_app.ps1"

pause
