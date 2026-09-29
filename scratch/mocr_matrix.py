# MangaOCR CPU speed matrix: one run tests every candidate on the same real crops.
# Why: extraction is now the bottleneck (第5巻 p1-30: MangaOCR ~0.5 s/crop, 232 crops = ~117 s,
# serialised by _MANGA_OCR_LOCK). Suspects: 1) decoder (autoregressive) dominates, so an ONNX
# encoder alone helps little; 2) batch size / torch thread count; 3) int8 dynamic quant of the
# decoder; 4) two OCR calls at once (instead of one lock) using the 12 cores better.
# Quality guard: exact-match text vs the production config (row 1).
# Run (project root, web app may stay closed):
#   python scratch\mocr_matrix.py "C:\Users\Work\Desktop\project\ori\図書館の大魔術師 第5巻.pdf" --pages 1-30
# ~5-8 min. Paste the whole table.
import argparse, json, os, pickle, sys, time, statistics
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pdf_translate"))
OUT = os.path.join(ROOT, "scratch", "mocr_matrix")
ENC_ONNX = os.path.join(ROOT, "pdf_translate", "data", "models", "onnx", "manga_ocr_encoder.onnx")


def collect_crops(pdf, pages, limit):
    """Same crops the extractor feeds MangaOCR: CTD boxes (+B18/B24 split, B22 dedupe), pad 8, 850 px page."""
    cache = os.path.join(OUT, "crops.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            crops = pickle.load(f)
        print(f"crops: {len(crops)} (cached {cache}; delete it to re-collect)")
        return crops[:limit]
    import fitz
    from PIL import Image
    from core import extractor as X
    det = X._get_comic_detector()
    doc = fitz.open(pdf)
    crops = []
    for p in pages:
        page = doc[p - 1]
        s = 850.0 / page.rect.height
        pix = page.get_pixmap(matrix=fitz.Matrix(s, s), alpha=False)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()
        seg = []
        boxes = X._detect_with_comic_detector(img, det, out_seg=seg)
        if seg:
            boxes = [q for b in boxes for q in X._split_box_by_seg(seg[0], b)]
        boxes = X._drop_contained_boxes(boxes)
        for b in boxes:
            x0, y0, x1, y1 = [int(v) for v in b[:4]]
            cx0, cy0 = max(0, x0 - 8), max(0, y0 - 8)
            cx1, cy1 = min(img.shape[1], x1 + 8), min(img.shape[0], y1 + 8)
            if cx1 - cx0 < 8 or cy1 - cy0 < 8:
                continue
            im = Image.fromarray(img[cy0:cy1, cx0:cx1])
            if im.width < 16 or im.height < 16:
                im = im.resize((max(im.width, 32), max(im.height, 32)), Image.LANCZOS)
            crops.append(im)
        print(f"  page {p}: {len(boxes)} boxes", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(crops, f)
    print(f"crops: {len(crops)}")
    return crops[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--pages", default="1-30")
    ap.add_argument("--limit", type=int, default=96, help="crops used per config")
    a = ap.parse_args()
    lo, hi = [int(v) for v in a.pages.split("-")]
    crops = collect_crops(a.pdf, range(lo, hi + 1), a.limit)

    import torch
    from manga_ocr import MangaOcr
    from manga_ocr.ocr import post_process
    from transformers.modeling_outputs import BaseModelOutput
    local = os.path.join(ROOT, "pdf_translate", "manga-ocr-base")
    mocr = MangaOcr(pretrained_model_name_or_path=local if os.path.exists(local) else "kha-white/manga-ocr-base",
                    force_cpu=True)
    model, tok = mocr.model.eval(), mocr.tokenizer
    default_threads = torch.get_num_threads()
    pix_all = [mocr._preprocess(im) for im in crops]

    def decode(out):
        return [post_process(tok.decode(r, skip_special_tokens=True)).strip() for r in out]

    def run_torch(batch, m=model, enc=None):
        texts = []
        for i in range(0, len(pix_all), batch):
            x = torch.stack(pix_all[i:i + batch])
            with torch.inference_mode():
                kw = {}
                if enc is not None:
                    kw["encoder_outputs"] = BaseModelOutput(last_hidden_state=torch.from_numpy(enc(x.numpy())))
                out = m.generate(x, max_new_tokens=64, max_length=None, **kw)
            texts += decode(out)
        return texts

    def run_encoder_only(batch):
        for i in range(0, len(pix_all), batch):
            with torch.inference_mode():
                model.encoder(pixel_values=torch.stack(pix_all[i:i + batch]))
        return None

    def onnx_encoder(threads):
        import onnxruntime as ort
        o = ort.SessionOptions()
        o.intra_op_num_threads = threads
        s = ort.InferenceSession(ENC_ONNX, sess_options=o, providers=["CPUExecutionProvider"])

        def enc(x):
            try:
                return s.run(["last_hidden_state"], {"pixel_values": x.astype(np.float32)})[0]
            except Exception:   # export may have a fixed batch of 1
                return np.concatenate([s.run(["last_hidden_state"], {"pixel_values": x[i:i + 1].astype(np.float32)})[0]
                                       for i in range(len(x))])
        return enc

    def quant_decoder():
        import copy
        q = copy.deepcopy(model)
        q.decoder = torch.quantization.quantize_dynamic(q.decoder, {torch.nn.Linear}, dtype=torch.qint8)
        return q

    def run_two_at_once(batch, threads_each):
        from concurrent.futures import ThreadPoolExecutor
        torch.set_num_threads(threads_each)
        chunks = [pix_all[i:i + batch] for i in range(0, len(pix_all), batch)]
        res = [None] * len(chunks)

        def work(k):
            with torch.inference_mode():
                res[k] = decode(model.generate(torch.stack(chunks[k]), max_new_tokens=64, max_length=None))
        with ThreadPoolExecutor(2) as ex:
            list(ex.map(work, range(len(chunks))))
        return [t for r in res for t in r]

    T = default_threads
    configs = [
        (f"torch b16 t{T} (production)", lambda: run_torch(16)),
        (f"torch b8 t{T}", lambda: run_torch(8)),
        (f"torch b32 t{T}", lambda: run_torch(32)),
        ("torch b16 t4", lambda: run_torch(16), 4),
        ("torch b16 t8", lambda: run_torch(16), 8),
        (f"encoder only b16 t{T} (share of total)", lambda: run_encoder_only(16)),
        (f"ONNX encoder t{T} + torch decoder b16", "onnx"),
        (f"torch b16 int8 decoder t{T}", "quant"),
        ("2 batches at once b16 t6 each", lambda: run_two_at_once(16, max(1, T // 2))),
    ]
    rows, base_txt, base_s = [], None, None
    for c in configs:
        name, fn = c[0], c[1]
        threads = c[2] if len(c) > 2 else T
        try:
            torch.set_num_threads(threads)
            if fn == "onnx":
                if not os.path.exists(ENC_ONNX):
                    raise FileNotFoundError(ENC_ONNX)
                enc = onnx_encoder(T)
                fn = (lambda e=enc: run_torch(16, enc=e))
            elif fn == "quant":
                qm = quant_decoder()
                fn = (lambda m=qm: run_torch(16, m=m))
            keep = pix_all
            pix_all[:] = keep[:16]
            fn()                                   # warm-up on 16 crops
            pix_all[:] = keep
            t0 = time.time()
            txt = fn()
            secs = time.time() - t0
        except Exception as e:
            rows.append({"name": name, "status": f"FAILED: {type(e).__name__}: {str(e)[:120]}"})
            print(rows[-1], flush=True)
            continue
        finally:
            torch.set_num_threads(T)
        if base_txt is None and txt is not None:
            base_txt, base_s = txt, secs
        same = None if txt is None else sum(x == y for x, y in zip(txt, base_txt))
        r = {"name": name, "status": "ok", "total_s": round(secs, 1), "per_crop_s": round(secs / len(pix_all), 3),
             "speedup": round(base_s / secs, 2), "identical": None if same is None else f"{same}/{len(pix_all)}"}
        if txt is not None and same is not None and same < len(pix_all):
            r["diff_sample"] = [(x, y) for x, y in zip(base_txt, txt) if x != y][:3]
        rows.append(r)
        print(r, flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(f"\ncrops={len(pix_all)} cpu_threads={T}")
    print(f"{'config':44s} {'status':8s} {'total':>7s} {'s/crop':>7s} {'x':>5s}  identical")
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['name']:44s} {r['status']}")
        else:
            print(f"{r['name']:44s} {'ok':8s} {r['total_s']:7.1f} {r['per_crop_s']:7.3f} {r['speedup']:5.2f}  {r['identical'] or '-'}")
    for r in rows:
        if r.get("diff_sample"):
            print(f"  diff {r['name']}: {r['diff_sample']}")


if __name__ == "__main__":
    main()
