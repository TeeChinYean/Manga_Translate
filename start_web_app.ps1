# ==============================================================================
# PDF 漫画翻译系统启动脚本 (PowerShell)
# ==============================================================================

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = "PDF 漫画翻译系统 [Port 8000]"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "   ⚡ PDF 漫画翻译与排版保真引擎 · Web 服务" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "正在检测大模型推理服务状态 (Port 18088 / 18089)..." -ForegroundColor Yellow
$llm18088 = Test-NetConnection -ComputerName 127.0.0.1 -Port 18088 -WarningAction SilentlyContinue -InformationLevel Quiet
$llm18089 = Test-NetConnection -ComputerName 127.0.0.1 -Port 18089 -WarningAction SilentlyContinue -InformationLevel Quiet

if ($llm18088) {
    Write-Host "✔ Turbovec RAG 网关服务已连接 (http://127.0.0.1:18088)" -ForegroundColor Green
} elseif ($llm18089) {
    Write-Host "✔ 本地 llama-server 推理服务已连接 (http://127.0.0.1:18089)" -ForegroundColor Green
} else {
    Write-Host "⚠️ 未检测到运行中的大模型服务，正在后台自动启动 Qwen 3.5 4B..." -ForegroundColor Cyan
    $ragDir = Join-Path $ScriptDir "..\qwen_turbovec_rag"
    $launcherPy = Join-Path $ragDir "app\llm_launcher.py"
    if (Test-Path $launcherPy) {
        Start-Process -FilePath "python" -ArgumentList "app\llm_launcher.py --model 1" -WorkingDirectory $ragDir -WindowStyle Minimized
        Write-Host "   正在等待大模型加载模型权重并绑定端口 (Port 18089)..." -ForegroundColor Yellow
        for ($i = 0; $i -lt 25; $i++) {
            Start-Sleep -Seconds 1
            $check = Test-NetConnection -ComputerName 127.0.0.1 -Port 18089 -WarningAction SilentlyContinue -InformationLevel Quiet
            if ($check) {
                Write-Host "✔ Qwen 3.5 4B 大模型服务已成功就绪！" -ForegroundColor Green
                break
            }
        }
    } else {
        Write-Host "   提示: 如需启用本地大模型，请确保 qwen_turbovec_rag 目录存在。" -ForegroundColor DarkYellow
    }
}

Write-Host ""
Write-Host "正在启动 PDF 翻译后端服务 (http://127.0.0.1:8000)..." -ForegroundColor Green
Write-Host "• 本地操作界面: http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host "• 浏览器将在 2 秒后自动打开..." -ForegroundColor Gray
Write-Host "--------------------------------------------------------" -ForegroundColor DarkGray
Write-Host ""

$browserJob = [powershell]::Create().AddScript({
    Start-Sleep -Seconds 2
    Start-Process "http://127.0.0.1:8000"
})
$browserJob.BeginInvoke() | Out-Null

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
Set-Location (Join-Path $ScriptDir "pdf_translate")

python main.py
