#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Is there a GPU at all?  One answer for the whole app, so a machine without a GPU runs everything
on the CPU (ONNX LaMa, CPU MangaOCR / detector / RapidOCR, no GPU child processes, no VRAM probes)
instead of trying each GPU path and failing over one by one.

Checked in this order (first hit wins):
  1. CPU_ONLY=1 / true / yes     -> no GPU (manual switch, works on any machine)
  2. torch.cuda.is_available()   -> GPU (live check, cheap; also what tests fake)
  3. nvidia-smi -L               -> GPU (NVIDIA driver present; result cached)
  4. Windows: a real video adapter + onnxruntime DirectML -> GPU (AMD / Intel; result cached)
Nothing found -> CPU.
"""

import os
import re
import subprocess
import threading

_LOCK = threading.Lock()
_PROBED = {}          # cached results of the slow probes: {"nvidia": name|None, "windows": name|None}

_FAKE_ADAPTERS = ("microsoft basic", "remote", "virtual", "parsec", "displaylink", "citrix", "vmware",
                  "hyper-v", "indirect")


def cpu_only_forced() -> bool:
    return os.getenv("CPU_ONLY", "").strip().lower() in ("1", "true", "yes", "on")


def _probe_nvidia_smi():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=3)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    m = re.search(r"GPU \d+:\s*([^(\n]+)", out.stdout or "")
    return m.group(1).strip() if m else None


def _probe_windows_adapter():
    """Name of a real (non-virtual) video adapter when onnxruntime can use DirectML, else None."""
    if os.name != "nt":
        return None
    try:
        import onnxruntime as ort
        if "DmlExecutionProvider" not in ort.get_available_providers():
            return None
        ps = "(Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }) -join '|'"
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=8)
    except Exception:
        return None
    for name in (out.stdout or "").strip().split("|"):
        name = name.strip()
        if name and not any(f in name.lower() for f in _FAKE_ADAPTERS):
            return name
    return None


def _cached(key, fn):
    with _LOCK:
        if key not in _PROBED:
            _PROBED[key] = fn()
        return _PROBED[key]


def _torch_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def gpu_info() -> dict:
    """{"gpu": bool, "source": str, "name": str}"""
    if cpu_only_forced():
        return {"gpu": False, "source": "CPU_ONLY=1", "name": ""}
    if _torch_cuda():
        return {"gpu": True, "source": "torch.cuda", "name": ""}
    name = _cached("nvidia", _probe_nvidia_smi)
    if name:
        return {"gpu": True, "source": "nvidia-smi", "name": name}
    name = _cached("windows", _probe_windows_adapter)
    if name:
        return {"gpu": True, "source": "windows+directml", "name": name}
    return {"gpu": False, "source": "no GPU detected", "name": ""}


def gpu_available() -> bool:
    return gpu_info()["gpu"]


def reset_cache() -> None:
    with _LOCK:
        _PROBED.clear()


def describe() -> str:
    i = gpu_info()
    if i["gpu"]:
        return f"GPU available ({i['source']}{': ' + i['name'] if i['name'] else ''})"
    return f"no GPU -> everything runs on the CPU ({i['source']})"
