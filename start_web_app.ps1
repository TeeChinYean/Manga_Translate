# ==============================================================================
# PDF 漫画翻译系统启动脚本 (PowerShell)
# 自动检测本地 Turbovec 大模型服务状态，启动 FastAPI Web 服务并打开浏览器
# ==============================================================================

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = "PDF 漫画翻译系统 - 运行中 [Port 8000]"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "   ⚡ PDF 漫画翻译与排版保真引擎 · Web 服务" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# 1. 检测本地 Turbovec Qwen 3.5 大模型服务
Write-Host "正在检测大模型推理服务状态 (Port 18088 / 18089)..." -ForegroundColor Yellow
$llm18088 = Test-NetConnection -ComputerName 127.0.0.1 -Port 18088 -WarningAction SilentlyContinue -InformationLevel Quiet
$llm18089 = Test-NetConnection -ComputerName 127.0.0.1 -Port 18089 -WarningAction SilentlyContinue -InformationLevel Quiet

if ($llm18088) {
    Write-Host "✔ Turbovec RAG 网关服务已连接 (http://127.0.0.1:18088)" -ForegroundColor Green
} elseif ($llm18089) {
    Write-Host "✔ 本地 llama-server 推理服务已连接 (http://127.0.0.1:18089)" -ForegroundColor Green
} else {
    Write-Host "⚠️ 未检测到 Turbovec 本地大模型服务 (Port 18088/18089)。" -ForegroundColor DarkYellow
    Write-Host "   提示: 如需使用 Qwen 3.5 4B 日漫汉化与对话润色功能，请先前往:" -ForegroundColor Gray
    Write-Host "   c:\Users\Work\Desktop\project\qwen_turbovec_rag 运行 start_full_system.bat" -ForegroundColor Cyan
    Write-Host "   (若仅测试页面排版或数字 PDF 提取，当前仍可正常使用)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "正在启动 PDF 翻译后端服务 (http://127.0.0.1:8000)..." -ForegroundColor Green
Write-Host "• 本地操作界面: http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host "• API 接口文档: http://127.0.0.1:8000/docs" -ForegroundColor Cyan
Write-Host "• 4GB 显存保护: 渲染并发已锁定为 1，杜绝 CUDA OOM" -ForegroundColor DarkGray
Write-Host "--------------------------------------------------------" -ForegroundColor DarkGray
Write-Host ""

# 2. 异步在 2 秒后打开默认浏览器
$browserJob = [powershell]::Create().AddScript({
    Start-Sleep -Seconds 2
    Start-Process "http://127.0.0.1:8000"
})
$browserJob.BeginInvoke() | Out-Null

# 3. 运行主程序
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
Set-Location (Join-Path $ScriptDir "pdf_translate")

try {
    python main.py
} catch {
    Write-Host ""
    Write-Host "❌ 服务异常停止: $($_.Exception.Message)" -ForegroundColor Red
}
