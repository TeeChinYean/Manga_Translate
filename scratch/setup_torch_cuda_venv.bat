@echo off
:: Separate test env for MangaOCR / LaMa on the GPU with PyTorch CUDA (your current Python has
:: torch +cpu). Does NOT touch the Python used by start_web_app.bat. Download ~3 GB.
setlocal
cd /d "%~dp0.."
if not exist .venv-torch (
    echo Creating .venv-torch ...
    python -m venv .venv-torch || goto :fail
)
call .venv-torch\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 || goto :fail
python -m pip install manga-ocr pymupdf opencv-python onnxruntime pillow numpy || goto :fail
python -m pip install simple-lama-inpainting --no-deps || goto :fail
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" || goto :fail
echo.
echo [OK] Now run:
echo    .venv-torch\Scripts\python scratch\gpu_ocr_lama_check.py
exit /b 0
:fail
echo [ERROR] setup failed - copy the messages above.
exit /b 1
