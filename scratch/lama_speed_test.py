# Why is one LaMa call ~8.5s on CPU? Two suspects:
#   1. graph_optimization_level = ORT_DISABLE_ALL (set for GPU FFC/DFT stability; CPU may not need it)
#   2. thread layout (LAMA_PARALLEL x LAMA_THREADS; current 2 x 3 = only 6 of 12 threads)
# This script times real 512x512 crops under each config and checks output vs the current
# production config (PSNR inside the mask; >= 40 dB = visually identical).
#
# Run from repo root (LLM may stay running, CPU only):
#   python scratch\lama_speed_test.py                     (第5巻.pdf page 1)
#   python scratch\lama_speed_test.py x.pdf --page 3 --runs 4
import argparse, os, sys, time, threading
import numpy as np, cv2, fitz
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "pdf_translate", "data", "models", "onnx", "lama.onnx")
LEVELS = {
    "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}


def make_inputs(pdf, page, n):
    doc = fitz.open(pdf)
    p = doc[page - 1]
    pix = p.get_pixmap(matrix=fitz.Matrix(1100 / p.rect.height, 1100 / p.rect.height), alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    doc.close()
    h, w = img.shape[:2]
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        y, x = int(rng.integers(0, max(1, h - 512))), int(rng.integers(0, max(1, w - 512)))
        crop = np.ascontiguousarray(img[y:y + 512, x:x + 512])
        crop = cv2.copyMakeBorder(crop, 0, 512 - crop.shape[0], 0, 512 - crop.shape[1], cv2.BORDER_REFLECT_101)
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        mask = (gray < 90).astype(np.uint8)  # dark strokes ~ text-like mask
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        out.append(((crop.astype(np.float32) / 255).transpose(2, 0, 1)[None],
                    mask.astype(np.float32)[None, None]))
    return out


def session(level, threads):
    o = ort.SessionOptions()
    o.graph_optimization_level = LEVELS[level]
    o.intra_op_num_threads = threads
    return ort.InferenceSession(MODEL, sess_options=o, providers=["CPUExecutionProvider"])


def bench(sess, inputs, parallel):
    sess.run(None, {"l_image_": inputs[0][0], "l_mask_": inputs[0][1]})  # warm-up
    outs = [None] * len(inputs)
    idx = iter(range(len(inputs)))
    lock = threading.Lock()

    def worker():
        while True:
            with lock:
                i = next(idx, None)
            if i is None:
                return
            outs[i] = sess.run(None, {"l_image_": inputs[i][0], "l_mask_": inputs[i][1]})[0]

    t = time.time()
    ts = [threading.Thread(target=worker) for _ in range(parallel)]
    [x.start() for x in ts]
    [x.join() for x in ts]
    return time.time() - t, outs


def psnr(a, b, m):
    sel = m[0, 0] > 0
    if not sel.any():
        return 99.0
    d = (a[0].transpose(1, 2, 0)[sel] - b[0].transpose(1, 2, 0)[sel]).astype(np.float64)
    mse = float((d ** 2).mean())
    return 99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?", default=os.path.join(ROOT, "第5巻.pdf"))
    ap.add_argument("--page", type=int, default=1)
    ap.add_argument("--runs", type=int, default=6, help="crops per config")
    a = ap.parse_args()
    cpu = os.cpu_count() or 4
    inputs = make_inputs(a.pdf, a.page, a.runs)
    configs = [("disable", 2, 3)]  # current production
    for lvl in ("basic", "extended", "all"):
        configs.append((lvl, 2, 3))
    for lvl in ("disable", "all"):
        for par, thr in ((1, cpu // 2), (1, cpu), (2, cpu // 4 * 1 or 1), (3, max(1, cpu // 3)), (2, cpu // 2)):
            if (lvl, par, thr) not in configs:
                configs.append((lvl, par, thr))
    print(f"cpu={cpu}  crops/config={a.runs}  ort={ort.__version__}", flush=True)
    base = None
    for lvl, par, thr in configs:
        try:
            s = session(lvl, thr)
            sec, outs = bench(s, inputs, par)
        except Exception as e:
            print(f"{lvl:9s} parallel={par} threads={thr:2d}  FAILED: {e}", flush=True)
            continue
        if base is None:
            base = (sec, outs)
        q = min(psnr(o, b, inp[1]) for o, b, inp in zip(outs, base[1], inputs))
        print(f"{lvl:9s} parallel={par} threads={thr:2d}  {sec:6.1f}s  {sec / a.runs:5.2f}s/crop  "
              f"x{base[0] / sec:4.2f}  minPSNR={q:5.1f}dB", flush=True)
        del s


if __name__ == "__main__":
    main()
