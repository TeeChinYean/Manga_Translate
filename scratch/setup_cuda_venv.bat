@echo off
:: Isolated test env for LaMa on CUDA. Does NOT touch the Python used by start_web_app.bat
:: (onnxruntime-gpu and onnxruntime-directml cannot be installed side by side).
:: Download ~1.5-2 GB (onnxruntime-gpu + CUDA 12 runtime + cuDNN 9 wheels from PyPI).
setlocal
cd /d "%~dp0.."
if not exist .venv-cuda (
    echo Creating .venv-cuda ...
    python -m venv .venv-cuda || goto :fail
)
call .venv-cuda\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install "onnxruntime-gpu[cuda,cudnn]>=1.21" numpy opencv-python pymupdf pillow || goto :fail
echo.
echo Checking CUDA provider ...
python -c "import onnxruntime as o; o.preload_dlls(); print(o.__version__, o.get_available_providers())" || goto :fail
echo.
echo [OK] Now stop the LLM, then run:
echo    .venv-cuda\Scripts\python scratch\lama_gpu_test.py --gpu-provider CUDAExecutionProvider
exit /b 0
:fail
echo [ERROR] setup failed - copy the messages above.
exit /b 1
