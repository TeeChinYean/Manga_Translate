# LaMa on GPU (DirectML/CUDA) vs CPU: speed, VRAM and output check on real manga crops.
# 1) STOP the LLM first (close the "Turbovec LLM Engine" window / llama-server), so VRAM is free.
# 2) Run from repo root:  python scratch\lama_gpu_test.py  (defaults: 第5巻.pdf, pages 1-10)
# Outputs a summary table and side-by-side images in scratch\lama_gpu_compare\ (original | CPU | GPU).
import argparse, os, sys, time, subprocess, logging
logging.disable(logging.WARNING)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pdf_translate"))
import numpy as np, cv2, fitz
from PIL import Image
import onnxruntime as ort
if hasattr(ort, "preload_dlls"):
    try:
        ort.preload_dlls()  # onnxruntime-gpu[cuda,cudnn]: load CUDA/cuDNN DLLs from the pip wheels
    except Exception as _e:
        print("preload_dlls failed:", _e)
from core import renderer as R
from core.extractor import _get_comic_detector, _detect_with_comic_detector

MODEL = os.path.join(ROOT, "pdf_translate", "data", "models", "onnx", "lama.onnx")
OUT = os.path.join(ROOT, "scratch", "lama_gpu_compare")


def nvidia_free_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


def make_session(provider):
    opts = ort.SessionOptions()
    if provider == "CPUExecutionProvider":
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # same as production
        opts.intra_op_num_threads = R.LAMA_THREADS
        return ort.InferenceSession(MODEL, sess_options=opts, providers=["CPUExecutionProvider"])
    if provider == "DmlExecutionProvider":
        opts.enable_mem_pattern = False  # required by DirectML
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if provider == "CUDAExecutionProvider":
        # Keep the CUDA arena small: this runs next to other GPU users on a 4 GB card
        cuda_opts = {"arena_extend_strategy": "kSameAsRequested", "cudnn_conv_algo_search": "HEURISTIC"}
        return ort.InferenceSession(MODEL, sess_options=opts,
                                    providers=[(provider, cuda_opts), "CPUExecutionProvider"])
    return ort.InferenceSession(MODEL, sess_options=opts, providers=[provider, "CPUExecutionProvider"])


def parse_pages(spec, n):
    pages = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            pages.update(range(int(a), int(b) + 1))
        elif part.strip():
            pages.add(int(part))
    return [p for p in sorted(pages) if 1 <= p <= n]


def collect_crops(pdf, pages, limit):
    """Real LaMa jobs exactly as production builds them: CTD boxes -> stroke masks -> regions."""
    doc, det, rend, jobs = fitz.open(pdf), _get_comic_detector(), R.PDFLayoutRenderer(), []
    for pn in pages:
        page = doc[pn - 1]
        img, scale = rend._rasterize(page)
        sd = 850.0 / page.rect.height
        small = cv2.resize(np.array(img), (int(page.rect.width * sd), int(page.rect.height * sd)),
                           interpolation=cv2.INTER_AREA)
        blocks = [{"id": i, "bbox": [v / sd for v in b], "color": (0, 0, 0)}
                  for i, b in enumerate(_detect_with_comic_detector(small, det))]
        prep = rend._build_masks(img, scale, {"page_num": pn, "blocks": blocks},
                                 {(pn, b["id"]): "译" for b in blocks})
        if prep["lama_mask"] is None:
            continue
        img_np = np.array(img)
        _, mask_bin = cv2.threshold(prep["lama_mask"], 10, 255, cv2.THRESH_BINARY)
        h, w = mask_bin.shape
        for (x0, y0, x1, y1) in R._lama_regions(mask_bin, w, h):
            x0, y0, x1, y1 = R._expand_to_window(x0, y0, x1, y1, w, h)
            m = mask_bin[y0:y1, x0:x1]
            if np.any(m):
                jobs.append((pn, img_np[y0:y1, x0:x1].copy(), m.copy()))
            if len(jobs) >= limit:
                return jobs
    return jobs


def run_all(session, jobs):
    R._lama_inpaint_crop(session, jobs[0][1], jobs[0][2])  # warm-up (kernel compile / allocs)
    outs, times = [], []
    for _, crop, m in jobs:
        t = time.perf_counter()
        outs.append(R._lama_inpaint_crop(session, crop, m))
        times.append(time.perf_counter() - t)
    return outs, times


def compare(a, b, m):
    sel = m > 0
    d = np.abs(a[sel].astype(np.float32) - b[sel].astype(np.float32))
    mse = float((d ** 2).mean()) if d.size else 0.0
    psnr = 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    return float(d.mean()) if d.size else 0.0, psnr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?", default=os.path.join(ROOT, "第5巻.pdf"))
    ap.add_argument("--pages", default="1-10")
    ap.add_argument("--limit", type=int, default=12, help="max crops to test")
    ap.add_argument("--gpu-provider", default=None, help="force DmlExecutionProvider / CUDAExecutionProvider")
    a = ap.parse_args()

    avail = ort.get_available_providers()
    gpu = a.gpu_provider or next((p for p in ("CUDAExecutionProvider", "DmlExecutionProvider") if p in avail), None)
    print("ORT providers:", avail)
    print("free VRAM before:", nvidia_free_mb(), "MB  (should be ~3900 with the LLM stopped)")
    if gpu is None:
        sys.exit("No GPU execution provider available in onnxruntime.")

    doc = fitz.open(a.pdf)
    jobs = collect_crops(a.pdf, parse_pages(a.pages, len(doc)), a.limit)
    print(f"collected {len(jobs)} real LaMa crops from {a.pdf}")
    if not jobs:
        sys.exit("No textured regions found on these pages; try other --pages.")

    cpu_out, cpu_t = run_all(make_session("CPUExecutionProvider"), jobs)
    free0 = nvidia_free_mb()
    try:
        gsess = make_session(gpu)
        print("GPU session providers:", gsess.get_providers())
        if gsess.get_providers()[0] != gpu:
            sys.exit(f"{gpu} could not be initialised (session fell back to CPU). "
                     "Check CUDA/cuDNN install messages above.")
        gpu_out, gpu_t = run_all(gsess, jobs)
    except Exception as e:
        sys.exit(f"GPU LaMa FAILED: {type(e).__name__}: {e}")
    free1 = nvidia_free_mb()

    os.makedirs(OUT, exist_ok=True)
    print(f"\n{'#':>2} page  CPU(s)  GPU(s)  meanAbsDiff  PSNR(dB)  verdict")
    bad = 0
    for i, ((pn, crop, m), co, go, ct, gt) in enumerate(zip(jobs, cpu_out, gpu_out, cpu_t, gpu_t)):
        diff, psnr = compare(co, go, m)
        broken = (not np.isfinite(go).all()) or psnr < 20
        bad += broken
        print(f"{i:>2} {pn:>4}  {ct:6.2f}  {gt:6.2f}  {diff:11.2f}  {psnr:8.1f}  {'CHECK IMAGE' if broken else 'ok'}")
        side = np.concatenate([crop, co, go], axis=1)
        Image.fromarray(side).save(os.path.join(OUT, f"crop_{i:02d}_p{pn}.png"))
    print(f"\nCPU total {sum(cpu_t):.1f}s  |  GPU total {sum(gpu_t):.1f}s  |  speed-up x{sum(cpu_t) / max(1e-6, sum(gpu_t)):.1f}")
    if free0 is not None and free1 is not None:
        print(f"GPU LaMa VRAM ~{free0 - free1:.0f} MB")
    print(f"{bad} crop(s) differ a lot from CPU (PSNR<20dB) -> look at {OUT}")
