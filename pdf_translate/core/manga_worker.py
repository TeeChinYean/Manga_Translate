"""
MangaOCR on CUDA (fp16) in a CHILD process (like core/lama_worker.py, BUG.md B29).

The web server process must not own a CUDA context (it keeps the LLM from being fully resident
in VRAM on a 4 GB card). This worker lives only during extraction in serial mode and exits
before translation, releasing all of its VRAM.

Protocol over a multiprocessing Pipe:
  child  -> ("ready", info) | ("err", message)        once, after loading
  parent -> ("ocr", [uint8 HxWx3 arrays])              child -> ("ok", [texts]) | ("err", message)
  parent -> None                                       exit
"""
import os
import sys

BATCH = int(os.getenv("MANGA_OCR_BATCH", "16"))


def worker_main(conn, model_ref: str, fp16: bool = True):
    try:
        import torch
        from PIL import Image
        from manga_ocr import MangaOcr
        from manga_ocr.ocr import post_process
        if not torch.cuda.is_available():
            conn.send(("err", "CUDA not available in worker"))
            return
        m = MangaOcr(pretrained_model_name_or_path=model_ref, force_cpu=False)
        if fp16:
            m.model.half()
        dtype = m.model.dtype
        conn.send(("ready", {"device": torch.cuda.get_device_name(0), "dtype": str(dtype)}))
    except Exception as e:  # pragma: no cover - machine dependent
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
            _cmd, arrays = msg
            texts = []
            for i in range(0, len(arrays), BATCH):
                ims = [Image.fromarray(a) for a in arrays[i:i + BATCH]]
                with torch.inference_mode():
                    x = torch.stack([m._preprocess(im) for im in ims]).to(m.model.device, dtype=dtype)
                    out = m.model.generate(x, max_new_tokens=64, max_length=None).cpu()
                texts += [post_process(m.tokenizer.decode(r, skip_special_tokens=True)).strip() for r in out]
            conn.send(("ok", texts))
        except Exception as e:
            conn.send(("err", f"{type(e).__name__}: {e}"))
    try:
        conn.close()
    except Exception:
        pass
