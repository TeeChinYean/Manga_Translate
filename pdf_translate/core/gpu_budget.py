#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU budget for extraction models: put an OCR/detection model on the GPU only when
there is enough free VRAM for it (the local LLM already lives there).

Free VRAM is probed in this order (first that works wins):
  1. torch.cuda.mem_get_info()          (CUDA builds of PyTorch)
  2. nvidia-smi --query-gpu=memory.free (NVIDIA driver present)
  3. Windows performance counters       (any vendor incl. DirectML: dedicated usage vs. total)
If nothing works the budget is "unknown" and models stay on CPU (safe default).

Env overrides:
  EXTRACT_DEVICE        auto (default) | cpu | gpu   (gpu = skip the VRAM check)
  EXTRACT_VRAM_RESERVE_MB   headroom left for LLM KV-cache growth / desktop (default 400)
"""

import os
import re
import time
import logging
import subprocess
import threading

logger = logging.getLogger(__name__)

# Approximate peak VRAM per model (weights + activations at our input sizes).
MODEL_VRAM_MB = {
    "comic_detector": 700,   # comic-text-detector.onnx @1024x1024
    "manga_ocr": 600,        # ViT encoder + BERT decoder, fp16 on CUDA (fp32 measured +791 MB)
    "paddle_ocr": 300,       # PP-OCRv4 det+cls+rec
}
DEFAULT_MODEL_VRAM_MB = dict(MODEL_VRAM_MB)
_MEASURED_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gpu_model_vram.json")
CALIBRATION_MARGIN = 1.2   # measured delta x1.2 (activations can peak higher on bigger pages)


def _load_measured():
    """Replace the default estimates with values measured on this machine (if any)."""
    try:
        import json
        with open(_MEASURED_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            if k in MODEL_VRAM_MB and isinstance(v, (int, float)) and 50 <= v <= 8000:
                MODEL_VRAM_MB[k] = int(v)
    except Exception:
        pass


def record_measured(model: str, delta_mb: float) -> bool:
    """
    Store a measured VRAM cost for `model` (free-before minus free-after its GPU load and
    first inference). Ignores implausible values caused by other processes (e.g. the LLM
    growing its KV cache at the same moment).
    """
    if model not in MODEL_VRAM_MB or delta_mb is None or not (30 <= delta_mb <= 6000):
        return False
    value = int(max(100, delta_mb * CALIBRATION_MARGIN))
    MODEL_VRAM_MB[model] = value
    try:
        import json
        data = {}
        if os.path.exists(_MEASURED_PATH):
            with open(_MEASURED_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        data[model] = value
        tmp = _MEASURED_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, _MEASURED_PATH)
    except Exception as e:
        logger.warning(f"[GPU Budget] could not save measured VRAM: {e}")
    logger.info(f"[GPU Budget] measured {model}: {int(delta_mb)}MB -> planning with {value}MB")
    return True


_load_measured()

# Placement priority: most GPU-benefit per MB first.
PRIORITY = ("comic_detector", "manga_ocr", "paddle_ocr")

_CACHE = {"t": 0.0, "free": None, "source": "none"}
_LOCK = threading.Lock()
_CACHE_TTL_S = 3.0


def _mode() -> str:
    m = os.getenv("EXTRACT_DEVICE", "auto").strip().lower()
    return m if m in ("auto", "cpu", "gpu") else "auto"


def _reserve_mb() -> int:
    try:
        return max(0, int(os.getenv("EXTRACT_VRAM_RESERVE_MB", "400")))
    except ValueError:
        return 400


def _probe_torch():
    import torch  # noqa: WPS433
    if not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info()
    return free / (1024 * 1024)


def _probe_nvidia_smi():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=3,
    )
    if out.returncode != 0:
        return None
    vals = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", out.stdout)]
    return max(vals) if vals else None


def _probe_windows_counters():
    if os.name != "nt":
        return None
    # WMI class names are not localized (Get-Counter paths are, e.g. on Chinese Windows).
    ps = (
        "$u=(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory "
        "-ErrorAction SilentlyContinue | Measure-Object -Property DedicatedUsage -Maximum).Maximum;"
        "$t=(Get-CimInstance Win32_VideoController | Measure-Object -Property AdapterRAM -Maximum).Maximum;"
        "Write-Output \"$u $t\""
    )
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                         capture_output=True, text=True, timeout=8)
    nums = re.findall(r"\d+(?:\.\d+)?", out.stdout)
    if len(nums) < 2:
        return None
    used, total = float(nums[0]), float(nums[1])
    if total <= 0:
        return None
    # Win32_VideoController.AdapterRAM is a uint32 and saturates at 4 GB on bigger cards.
    return max(0.0, (total - used) / (1024 * 1024))


def free_vram_mb(force: bool = False):
    """Return (free_mb or None, source). Cached for a few seconds."""
    with _LOCK:
        if not force and time.time() - _CACHE["t"] < _CACHE_TTL_S:
            return _CACHE["free"], _CACHE["source"]
        free, source = None, "none"
        for name, fn in (("torch.cuda", _probe_torch), ("nvidia-smi", _probe_nvidia_smi),
                         ("windows-counters", _probe_windows_counters)):
            try:
                v = fn()
            except Exception:
                v = None
            if v is not None:
                free, source = float(v), name
                break
        _CACHE.update(t=time.time(), free=free, source=source)
        return free, source


def plan_placement(capable: dict, free_mb=None, resident=(), source: str = "override") -> dict:
    """
    Decide GPU/CPU per model.
      capable:  {model_name: bool} — whether a GPU backend exists for that model.
      free_mb:  override the probe (tests).
      resident: models already loaded on the GPU. Their VRAM is already "used", so they
                stay on the GPU as long as free VRAM >= reserve/2 (hysteresis, no flapping).
    Returns {model_name: "gpu" | "cpu"} plus "_reason" and "_free_mb" keys.
    """
    mode = _mode()
    plan = {m: "cpu" for m in capable}
    if mode == "cpu":
        plan.update(_reason="EXTRACT_DEVICE=cpu", _free_mb=None)
        return plan
    if mode == "gpu":
        plan.update({m: "gpu" for m, ok in capable.items() if ok})
        plan.update(_reason="EXTRACT_DEVICE=gpu (VRAM check skipped)", _free_mb=None)
        return plan

    if free_mb is None:
        free_mb, source = free_vram_mb()
    if free_mb is None:
        plan.update(_reason="free VRAM unknown -> CPU", _free_mb=None)
        return plan

    budget = free_mb - _reserve_mb()
    notes = []
    for m in PRIORITY:
        if m not in capable:
            continue
        need = MODEL_VRAM_MB.get(m, 500)
        if not capable[m]:
            notes.append(f"{m}:cpu(no GPU backend)")
        elif m in resident:
            if free_mb >= _reserve_mb() / 2:
                plan[m] = "gpu"
                notes.append(f"{m}:gpu(resident)")
            else:
                notes.append(f"{m}:cpu(resident but only {int(free_mb)}MB free)")
        elif budget >= need:
            plan[m] = "gpu"
            budget -= need
            notes.append(f"{m}:gpu(-{need}MB)")
        else:
            notes.append(f"{m}:cpu(need {need}MB, left {int(budget)}MB)")
    plan.update(_reason=f"free={int(free_mb)}MB via {source}, reserve={_reserve_mb()}MB; " + ", ".join(notes),
                _free_mb=free_mb)
    return plan


def is_gpu_oom(exc: BaseException) -> bool:
    """Heuristic: does this exception look like a GPU out-of-memory / device-lost error?"""
    msg = f"{type(exc).__name__}: {exc}".lower()
    keys = ("out of memory", "outofmemory", "cuda error", "cudnn", "e_outofmemory",
            "887a0005", "887a0006", "dml", "directml", "failed to allocate", "device removed")
    return any(k in msg for k in keys)
