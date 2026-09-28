# Check what GPU backends the extraction models can use and how much VRAM is free.
# Run with the same Python that start_web_app.bat uses (repo root):  python scratch\gpu_diag.py
import os, sys, logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pdf_translate"))
from core import gpu_budget

print("python:", sys.executable)
try:
    import onnxruntime as ort
    print("onnxruntime", ort.__version__, "providers:", ort.get_available_providers())
except Exception as e:
    print("onnxruntime: NOT available", e)
try:
    import torch
    print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
except Exception as e:
    print("torch: NOT available", e)
for name, fn in (("torch.cuda", gpu_budget._probe_torch), ("nvidia-smi", gpu_budget._probe_nvidia_smi),
                 ("windows-counters", gpu_budget._probe_windows_counters)):
    try:
        print(f"probe {name:<17}:", fn())
    except Exception as e:
        print(f"probe {name:<17}: error {e}")
from core.extractor import prepare_extract_devices
print("placement:", prepare_extract_devices())
