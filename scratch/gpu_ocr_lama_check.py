# Can MangaOCR and LaMa run on the GPU (RTX 3050 Laptop, 4 GB) next to / instead of the LLM?
# Based on the two snippets you sent (OCr.txt = MangaOCR batched on CUDA, laMa.txt = simple-lama
# on CUDA), but on REAL crops / windows from a manga page, compared with what production uses now
# (MangaOCR on CPU, LaMa ONNX on CPU), with a quality check and VRAM numbers.
#
# Needs PyTorch CUDA: run scratch\setup_torch_cuda_venv.bat once, then:
#   .venv-torch\Scripts\python scratch\gpu_ocr_lama_check.py                  (第5巻.pdf pages 1-3)
#   .venv-torch\Scripts\python scratch\gpu_ocr_lama_check.py x.pdf --pages 1-5
# Run it twice: once with the LLM running (is there room next to it?) and once with the LLM
# stopped (how fast is the GPU when it has the card to itself?).
import argparse, os, subprocess, sys, time, logging
logging.disable(logging.WARNING)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pdf_translate"))
import numpy as np, cv2, fitz
from PIL import Image


def vram():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        used, total = (int(v) for v in out.split(","))
        return used, total
    except Exception:
        return None, None


def llm_running():
    import urllib.request
    try:
        urllib.request.urlopen("http://127.0.0.1:18089/health", timeout=1.5)
        return True
    except Exception:
        return False


def page_images(pdf, pages, height=850):
    doc = fitz.open(pdf)
    for pn in pages:
        page = doc[pn - 1]
        s = height / page.rect.height
        pix = page.get_pixmap(matrix=fitz.Matrix(s, s), alpha=False)
        yield pn, np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()


def text_boxes(img):
    from core.extractor import _get_comic_detector, _detect_with_comic_detector
    det = _get_comic_detector()
    return [tuple(int(v) for v in b) for b in _detect_with_comic_detector(img, det)] if det else []


def section(t):
    print("\n== " + t, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?", default=os.path.join(ROOT, "第5巻.pdf"))
    ap.add_argument("--pages", default="1-3")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.pages.split("-"))

    import torch
    section("environment")
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}"
          + (f"  {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    used, total = vram()
    print(f"LLM on :18089 running = {llm_running()}   VRAM used {used} / {total} MB")
    if not torch.cuda.is_available():
        sys.exit("CUDA not available in this Python -> run scratch\\setup_torch_cuda_venv.bat and use .venv-torch")
    dev = torch.device("cuda")

    # ---------- real crops + LaMa windows from the PDF ----------
    crops, windows = [], []
    for pn, img in page_images(a.pdf, range(lo, hi + 1)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        for x0, y0, x1, y1 in text_boxes(img):
            c = img[max(0, y0 - 8):y1 + 8, max(0, x0 - 8):x1 + 8]
            if c.size:
                crops.append(Image.fromarray(c))
            # LaMa window: 512 around the box on the page rendered at 1600 px like production
            m = np.zeros(gray.shape, np.uint8)
            m[y0:y1, x0:x1] = (gray[y0:y1, x0:x1] < 120).astype(np.uint8) * 255
            m = cv2.dilate(m, np.ones((5, 5), np.uint8))
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            h, w = gray.shape
            s = 256
            X0, Y0 = max(0, min(w - 2 * s, cx - s)), max(0, min(h - 2 * s, cy - s))
            win, wm = img[Y0:Y0 + 2 * s, X0:X0 + 2 * s], m[Y0:Y0 + 2 * s, X0:X0 + 2 * s]
            if win.shape[:2] == (512, 512) and wm.any():
                windows.append((win.copy(), wm.copy()))
    print(f"{len(crops)} OCR crops, {len(windows)} LaMa 512x512 windows from pages {lo}-{hi}")

    # ---------- MangaOCR: CPU (production) vs CUDA ----------
    section("MangaOCR  (OCr.txt approach: processor -> generate in batches)")
    from manga_ocr import MangaOcr
    from manga_ocr.ocr import post_process
    local = os.path.join(ROOT, "pdf_translate", "manga-ocr-base")
    ref = local if os.path.exists(local) else "kha-white/manga-ocr-base"

    @torch.inference_mode()
    def run_batch(m, ims, bs, device):
        out = []
        for i in range(0, len(ims), bs):
            imgs = [c.convert("L").convert("RGB") for c in ims[i:i + bs]]
            pv = m.processor(imgs, return_tensors="pt").pixel_values.to(device)
            ids = m.model.generate(pv, max_length=300)
            out += [post_process(t) for t in m.tokenizer.batch_decode(ids.cpu(), skip_special_tokens=True)]
        return out

    cpu = MangaOcr(pretrained_model_name_or_path=ref, force_cpu=True)
    run_batch(cpu, crops[:2], 2, "cpu")
    t = time.perf_counter(); base = run_batch(cpu, crops, a.batch, "cpu"); tc = time.perf_counter() - t
    print(f"CPU  batch {a.batch}: {tc:6.1f}s  ({tc / max(1, len(crops)):.2f}s/crop)   <- production")
    del cpu
    before = vram()[0]
    try:
        gpu = MangaOcr(pretrained_model_name_or_path=ref, force_cpu=True)
        gpu.model.to(dev).eval()
        torch.cuda.reset_peak_memory_stats()
        run_batch(gpu, crops[:2], 2, dev)
        torch.cuda.synchronize(); t = time.perf_counter()
        got = run_batch(gpu, crops, a.batch, dev)
        torch.cuda.synchronize(); tg = time.perf_counter() - t
        same = sum(g == b for g, b in zip(got, base))
        print(f"CUDA batch {a.batch}: {tg:6.1f}s  ({tg / max(1, len(crops)):.2f}s/crop)  x{tc / tg:.1f}  "
              f"identical text {same}/{len(crops)}")
        print(f"     VRAM: torch peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MB, "
              f"nvidia-smi {before} -> {vram()[0]} MB")
        gpu.model.half()
        pv_ok = True
        try:
            run_batch_fp16(gpu, crops[:2], 2, dev, post_process)
            torch.cuda.synchronize(); t = time.perf_counter()
            got16 = run_batch_fp16(gpu, crops, a.batch, dev, post_process)
            torch.cuda.synchronize()
        except Exception as e:
            pv_ok = False
            print(f"CUDA fp16: failed ({e})")
        if pv_ok:
            t16 = time.perf_counter() - t
            same16 = sum(g == b for g, b in zip(got16, base))
            print(f"CUDA fp16  : {t16:6.1f}s  x{tc / t16:.1f}  identical text {same16}/{len(crops)}  "
                  f"(half the weights VRAM)")
        del gpu
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        print("CUDA: OUT OF MEMORY (the LLM leaves too little VRAM) -> stop the LLM and run again")

    # ---------- LaMa: ONNX CPU (production) vs simple-lama (torch) CUDA ----------
    section("LaMa  (laMa.txt approach: simple-lama big-lama on CUDA)")
    if not windows:
        print("no windows"); return
    from core import renderer as R
    sess = R._get_lama_session()
    if sess is None:
        print("lama.onnx not found -> skip production comparison")
    else:
        R._lama_inpaint_crop(sess, windows[0][0], windows[0][1])
        t = time.perf_counter()
        ref_out = [R._lama_inpaint_crop(sess, w, m) for w, m in windows]
        tl = time.perf_counter() - t
        print(f"ONNX CPU : {tl:6.1f}s  ({tl / len(windows):.2f}s/window, 1 at a time)   <- production")
    try:
        from simple_lama_inpainting import SimpleLama
    except Exception as e:
        print(f"simple-lama not installed ({e}) -> run setup_torch_cuda_venv.bat"); return
    before = vram()[0]
    try:
        lama = SimpleLama(device=dev)            # downloads big-lama weights on first run
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            lama(Image.fromarray(windows[0][0]), Image.fromarray(windows[0][1]))
            torch.cuda.synchronize(); t = time.perf_counter()
            outs = [np.array(lama(Image.fromarray(w), Image.fromarray(m)))[:512, :512] for w, m in windows]
            torch.cuda.synchronize(); tg = time.perf_counter() - t
        print(f"CUDA     : {tg:6.1f}s  ({tg / len(windows):.2f}s/window)"
              + (f"  x{tl / tg:.1f}" if sess is not None else ""))
        print(f"     VRAM: torch peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MB, "
              f"nvidia-smi {before} -> {vram()[0]} MB")
        if sess is not None:
            ps = []
            for (w, m), o, r in zip(windows, outs, ref_out):
                sel = m > 0
                d = (o[sel].astype(float) - r[sel].astype(float))
                mse = float((d ** 2).mean()) if d.size else 0.0
                ps.append(99.0 if mse < 1e-9 else 10 * np.log10(255 ** 2 / mse))
            print(f"     PSNR vs production (inside mask): min {min(ps):.1f} dB, median {np.median(ps):.1f} dB"
                  "  (same model family; >=30 dB = very close, compare images if lower)")
    except torch.cuda.OutOfMemoryError:
        print("CUDA: OUT OF MEMORY (the LLM leaves too little VRAM) -> stop the LLM and run again")

    section("summary")
    print("Paste this whole output back. What decides it: CUDA speed-up per stage, identical text /")
    print("PSNR, and whether it fits NEXT TO the LLM (VRAM) or only when the LLM is stopped.")


@__import__("torch").inference_mode()
def run_batch_fp16(m, ims, bs, device, post_process):
    import torch
    out = []
    for i in range(0, len(ims), bs):
        imgs = [c.convert("L").convert("RGB") for c in ims[i:i + bs]]
        pv = m.processor(imgs, return_tensors="pt").pixel_values.to(device, dtype=torch.float16)
        ids = m.model.generate(pv, max_length=300)
        out += [post_process(t) for t in m.tokenizer.batch_decode(ids.cpu(), skip_special_tokens=True)]
    return out


if __name__ == "__main__":
    main()
