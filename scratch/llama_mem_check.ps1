# Why does com.docker.llama-server.exe use so much RAM?  Prints, for every running llama-server:
#   1. memory: Private (committed, what Task Manager "Memory" shows) vs Working Set (incl. mmap'd model file)
#   2. the exact command line it was started with (who launched it, which flags)
#   3. whether this llama.cpp build supports the RAM-saving flags we may add
#   4. GPU memory in use (nvidia-smi)
# Run (PowerShell):  powershell -ExecutionPolicy Bypass -File scratch\llama_mem_check.ps1
$ErrorActionPreference = "SilentlyContinue"
$procs = Get-CimInstance Win32_Process | Where-Object { $_.Name -like "*llama-server*" }
if (-not $procs) { Write-Host "no llama-server process running"; exit 0 }
foreach ($p in $procs) {
    $gp = Get-Process -Id $p.ProcessId
    "{0}  PID {1}" -f $p.Name, $p.ProcessId
    "  Private (commit) : {0,8:N0} MB" -f ($gp.PrivateMemorySize64 / 1MB)
    "  Working set      : {0,8:N0} MB" -f ($gp.WorkingSet64 / 1MB)
    "  Peak working set : {0,8:N0} MB" -f ($gp.PeakWorkingSet64 / 1MB)
    "  Command line     : {0}" -f $p.CommandLine
    $exe = $p.ExecutablePath
}
""
"Flags supported by this build (empty = not supported):"
& $exe --help 2>&1 | Select-String -Pattern "no-mmap|mlock|cache-ram|ctx-checkpoints|swa-checkpoints|cache-reuse|kv-unified|--fit" | ForEach-Object { "  " + $_.Line.Trim() }
""
"GPU:"
& nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
