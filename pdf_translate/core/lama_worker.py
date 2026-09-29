"""
LaMa (big-lama TorchScript) on CUDA in a CHILD process (BUG.md B29).

Why a child process: on a 4 GB card any CUDA context inside the web server process keeps the
LLM (llama-server, Vulkan) from being fully resident in VRAM -> translation 4x slower. A CUDA
context cannot be destroyed in-process, but it disappears when the process exits. So rendering
starts this worker, streams 512x512 windows to it, and stops it when the job is done.

Protocol over a multiprocessing Pipe:
  parent -> ("run", image[1,3,H,W] float32 0..1, mask[1,1,H,W] float32 {0,1})
  child  -> ("ok", out[1,3,H,W] float32 0..255)  |  ("err", message)
  parent -> None                                   (exit)
The child first answers ("ready", info) or ("err", message) after loading the model.
"""
import os
import sys


def worker_main(conn, model_path: str):
    try:
        import numpy as np
        import torch
        if not torch.cuda.is_available():
            conn.send(("err", "CUDA not available in worker"))
            return
        dev = torch.device("cuda")
        model = torch.jit.load(model_path, map_location=dev).eval()
        conn.send(("ready", {"device": torch.cuda.get_device_name(0), "torch": torch.__version__}))
    except Exception as e:  # pragma: no cover - depends on the machine
        try:
            conn.send(("err", f"{type(e).__name__}: {e}"))
        finally:
            return
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        if msg is None:
            break
        try:
            _cmd, img, mask = msg
            with torch.inference_mode():
                out = model(torch.from_numpy(img).to(dev), torch.from_numpy(mask).to(dev))
                res = (out.clamp(0, 1) * 255.0).float().cpu().numpy()
            conn.send(("ok", res))
        except Exception as e:
            conn.send(("err", f"{type(e).__name__}: {e}"))
    try:
        conn.close()
    except Exception:
        pass


if __name__ == "__main__":  # manual smoke test: python core/lama_worker.py path/to/big-lama.pt
    import multiprocessing as mp
    a, b = mp.Pipe()
    p = mp.get_context("spawn").Process(target=worker_main, args=(b, sys.argv[1]), daemon=True)
    p.start()
    print(a.recv())
    a.send(None)
    p.join(10)
