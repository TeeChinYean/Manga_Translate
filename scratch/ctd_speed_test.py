# Can extraction run pages truly in parallel? Today CTD (comic-text-detector) runs under a
# global lock with default ORT threads, so 2 extraction workers still detect one page at a time.
# Same idea as LaMa (3 runs x 4 threads was x1.70): time CTD under parallel x threads layouts
# on real pages and check the boxes are identical to the current production result.
#
# Run from repo root (CPU only, LLM may keep running):
#   python scratch\ctd_speed_test.py                  (第5巻.pdf pages 1-12)
#   python scratch\ctd_speed_test.py x.pdf --pages 1-12
import argparse, os, sys, time, threading, contextlib, logging
logging.disable(logging.WARNING)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pdf_translate"))
import numpy as np, fitz
import onnxruntime as ort
from core import extractor as ex

MODEL = os.path.join(ROOT, "pdf_translate", "data", "models", "onnx", "comic-text-detector.onnx")


def load_pages(pdf, spec):
    a, b = (int(x) for x in spec.split("-"))
    doc = fitz.open(pdf)
    out = []
    for p in range(a, min(b, len(doc)) + 1):
        page = doc[p - 1]
        s = 850.0 / max(1.0, page.rect.height)  # same as _extract_single_page
        pix = page.get_pixmap(matrix=fitz.Matrix(s, s), alpha=False)
        out.append(np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy())
    doc.close()
    return out


def session(threads):
    o = ort.SessionOptions()
    if threads:
        o.intra_op_num_threads = threads
    return ort.InferenceSession(MODEL, sess_options=o, providers=["CPUExecutionProvider"])


def bench(sess, imgs, parallel):
    ex._detect_with_comic_detector(imgs[0], sess)  # warm-up
    outs = [None] * len(imgs)
    it = iter(range(len(imgs)))
    lock = threading.Lock()

    def worker():
        while True:
            with lock:
                i = next(it, None)
            if i is None:
                return
            outs[i] = [tuple(int(v) for v in b) for b in ex._detect_with_comic_detector(imgs[i], sess)]

    t = time.time()
    ts = [threading.Thread(target=worker) for _ in range(parallel)]
    [x.start() for x in ts]
    [x.join() for x in ts]
    return time.time() - t, outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?", default=os.path.join(ROOT, "第5巻.pdf"))
    ap.add_argument("--pages", default="1-12")
    a = ap.parse_args()
    cpu = os.cpu_count() or 4
    imgs = load_pages(a.pdf, a.pages)
    # Bypass the production lock so runs can overlap (ORT Run is thread-safe)
    ex._COMIC_DETECTOR_LOCK = contextlib.nullcontext() if False else threading.RLock()
    configs = [(1, 0, True)]  # production: 1 at a time (lock), default threads
    for par, thr in ((1, cpu), (1, cpu // 2), (2, cpu // 2), (3, cpu // 3), (4, cpu // 4), (6, cpu // 6), (2, cpu // 4)):
        configs.append((par, max(1, thr), False))
    print(f"cpu={cpu} pages={len(imgs)} ort={ort.__version__}", flush=True)
    base = None
    for par, thr, locked in configs:
        ex._COMIC_DETECTOR_LOCK = threading.Lock() if locked else contextlib.nullcontext()
        try:
            sec, outs = bench(session(thr), imgs, par)
        except Exception as e:
            print(f"parallel={par} threads={thr}  FAILED: {e}", flush=True)
            continue
        if base is None:
            base = (sec, outs)
        same = sum(o == b for o, b in zip(outs, base[1]))
        label = "default" if thr == 0 else str(thr)
        print(f"parallel={par} threads={label:>7}  {sec:6.1f}s  {sec / len(imgs):5.2f}s/page  "
              f"x{base[0] / sec:4.2f}  boxes identical {same}/{len(imgs)}", flush=True)


if __name__ == "__main__":
    main()
