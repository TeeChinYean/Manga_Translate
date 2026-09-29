# ONE run that tests every LaMa variant we may use, so nothing has to be re-run by hand:
#   models    : lama.onnx (current) | lama_fp32.onnx (Carve re-export) | lama-manga.onnx (manga-trained)
#               | big-lama.pt (torch, current GPU path)
#   runtimes  : ONNX CPU (threads 4 / 12, graph opt off / all) | DirectML | CUDA EP (onnxruntime-gpu)
#               | torch CUDA fp32 / fp16
#   Pythons   : this Python (onnxruntime-directml + CUDA torch), .venv-cuda (onnxruntime-gpu),
#               .venv-torch — each config runs in its OWN child process with a timeout, so a crash
#               (887A0005), a hang (280 s CUDA warm-up) or an OOM only marks that one row as failed.
# Output: a table (load s, s/window, peak VRAM, PSNR vs production), scratch\lama_matrix\results.json,
#         comparison sheets scratch\lama_matrix\sheet_*.jpg (one column per model's best config).
#
# STOP the LLM first (its VRAM would distort the numbers and a GPU test can evict it):
#   taskkill /F /IM com.docker.llama-server.exe
#   python scratch\lama_matrix.py                      (第5巻.pdf pages 1-4, 8 windows, ~10-15 min)
#   python scratch\lama_matrix.py x.pdf --pages 3-8 --only manga   (filter configs by substring)
import argparse, json, os, subprocess, sys, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "scratch", "lama_matrix")
ONNX_DIR = os.path.join(ROOT, "pdf_translate", "data", "models", "onnx")
BIG_LAMA = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints", "big-lama.pt")
VENV_CUDA = os.path.join(ROOT, ".venv-cuda", "Scripts", "python.exe")
VENV_TORCH = os.path.join(ROOT, ".venv-torch", "Scripts", "python.exe")
TIMEOUT_S = 420          # per config (load + warm-up + all windows)


def configs():
    me = sys.executable
    cuda_py = VENV_CUDA if os.path.exists(VENV_CUDA) else None
    torch_py = me
    out = []
    for m in ("lama.onnx", "lama_fp32.onnx", "lama-manga.onnx"):
        if not os.path.exists(os.path.join(ONNX_DIR, m)):
            continue
        tag = m.replace(".onnx", "")
        out += [
            dict(name=f"{tag} | CPU t4 opt=off", py=me, kind="onnx", model=m, ep="CPU", threads=4, opt="off"),
            dict(name=f"{tag} | CPU t12 opt=off", py=me, kind="onnx", model=m, ep="CPU", threads=12, opt="off"),
            dict(name=f"{tag} | CPU t4 opt=all", py=me, kind="onnx", model=m, ep="CPU", threads=4, opt="all"),
            dict(name=f"{tag} | DirectML opt=off", py=me, kind="onnx", model=m, ep="DML", opt="off"),
            dict(name=f"{tag} | DirectML opt=all", py=me, kind="onnx", model=m, ep="DML", opt="all"),
        ]
        if cuda_py:
            out += [
                dict(name=f"{tag} | CUDA opt=off", py=cuda_py, kind="onnx", model=m, ep="CUDA", opt="off"),
                dict(name=f"{tag} | CUDA opt=all", py=cuda_py, kind="onnx", model=m, ep="CUDA", opt="all"),
            ]
    if os.path.exists(BIG_LAMA):
        out += [
            dict(name="big-lama.pt | torch CUDA fp32", py=torch_py, kind="torch", model=BIG_LAMA, fp16=False),
            dict(name="big-lama.pt | torch CUDA fp16", py=torch_py, kind="torch", model=BIG_LAMA, fp16=True),
        ]
    return out


# ───────────────────────────── windows (real pages) ─────────────────────────────
def make_windows(pdf, lo, hi, path, limit=8):
    sys.path.insert(0, os.path.join(ROOT, "pdf_translate"))
    import numpy as np, cv2, fitz
    from core.extractor import _get_comic_detector, _detect_with_comic_detector
    det = _get_comic_detector()
    doc = fitz.open(pdf)
    imgs, masks = [], []
    for pn in range(lo, hi + 1):
        page = doc[pn - 1]
        s = 1600.0 / page.rect.height                          # production render height
        pix = page.get_pixmap(matrix=fitz.Matrix(s, s), alpha=False)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()
        small = cv2.resize(img, (int(img.shape[1] * 850 / img.shape[0]), 850))
        k = img.shape[0] / 850.0
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        for b in _detect_with_comic_detector(small, det):
            x0, y0, x1, y1 = (int(v * k) for v in b)
            m = np.zeros(gray.shape, np.uint8)
            m[y0:y1, x0:x1] = (gray[y0:y1, x0:x1] < 140).astype(np.uint8) * 255
            m = cv2.dilate(m, np.ones((7, 7), np.uint8))
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            X0 = max(0, min(img.shape[1] - 512, cx - 256)); Y0 = max(0, min(img.shape[0] - 512, cy - 256))
            w, wm = img[Y0:Y0 + 512, X0:X0 + 512], m[Y0:Y0 + 512, X0:X0 + 512]
            if w.shape[:2] == (512, 512) and wm.any() and len(imgs) < limit:
                imgs.append(w); masks.append(wm)
    np.savez_compressed(path, imgs=np.stack(imgs), masks=np.stack(masks))
    return len(imgs)


# ───────────────────────────── child: run one config ────────────────────────────
def child(cfg_json, win_path, out_path):
    import numpy as np
    cfg = json.loads(cfg_json)
    d = np.load(win_path)
    imgs, masks = d["imgs"], d["masks"]
    t0 = time.time()
    if cfg["kind"] == "onnx":
        import onnxruntime as ort
        if cfg["ep"] == "CUDA" and hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        o = ort.SessionOptions()
        o.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_ENABLE_ALL if cfg.get("opt") == "all"
                                      else ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
        if cfg["ep"] == "CPU":
            o.intra_op_num_threads = cfg.get("threads", 4)
            prov = ["CPUExecutionProvider"]
        elif cfg["ep"] == "DML":
            o.enable_mem_pattern = False
            o.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            prov = ["DmlExecutionProvider"]
        else:
            prov = ["CUDAExecutionProvider"]
        if prov[0] not in ort.get_available_providers():
            print(json.dumps({"error": f"{prov[0]} not available (onnxruntime {ort.__version__})"})); return
        sess = ort.InferenceSession(os.path.join(ONNX_DIR, cfg["model"]), sess_options=o, providers=prov)
        used = sess.get_providers()[0]
        ins = sess.get_inputs()
        img_in = next(i for i in ins if (i.shape[1] if len(i.shape) > 1 else 0) == 3)
        msk_in = next(i for i in ins if i is not img_in)

        def run(im, mk):
            x = (im.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
            m = (mk > 0).astype(np.float32)[None, None]
            x = x * (1.0 - m)
            y = sess.run(None, {img_in.name: x, msk_in.name: m})[0][0]
            y = y.transpose(1, 2, 0)
            if y.max() <= 1.5:
                y = y * 255.0
            return np.clip(y, 0, 255).astype(np.uint8)
        info = {"provider_used": used, "inputs": [(i.name, i.shape) for i in ins]}
    else:
        import torch
        dev = torch.device("cuda")
        model = torch.jit.load(cfg["model"], map_location=dev).eval()
        if cfg.get("fp16"):
            model = model.half()
        dt = torch.float16 if cfg.get("fp16") else torch.float32

        def run(im, mk):
            with torch.inference_mode():
                x = torch.from_numpy((im.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]).to(dev, dt)
                m = torch.from_numpy((mk > 0).astype(np.float32)[None, None]).to(dev, dt)
                y = model(x, m)[0].float().clamp(0, 1).mul(255).permute(1, 2, 0).cpu().numpy()
            return y.astype(np.uint8)
        info = {"provider_used": f"torch {torch.__version__} {'fp16' if cfg.get('fp16') else 'fp32'}"}
    first = run(imgs[0], masks[0])            # warm-up (the 280 s CUDA case shows up here)
    load_s = time.time() - t0
    outs, t1 = [], time.time()
    for im, mk in zip(imgs, masks):
        outs.append(run(im, mk))
    per = (time.time() - t1) / len(imgs)
    np.savez_compressed(out_path, outs=np.stack(outs))
    print(json.dumps(dict(info, load_s=round(load_s, 1), per_window_s=round(per, 3))))


# ───────────────────────────── master ───────────────────────────────────────────
def vram_used():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


def psnr_in_mask(a, b, m):
    import numpy as np
    sel = m > 0
    d = a[sel].astype(float) - b[sel].astype(float)
    mse = float((d ** 2).mean()) if d.size else 0.0
    return 99.0 if mse < 1e-9 else 10 * np.log10(255 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?", default=os.path.join(ROOT, "第5巻.pdf"))
    ap.add_argument("--pages", default="1-4")
    ap.add_argument("--only", default="", help="run only configs whose name contains this text")
    ap.add_argument("--windows", type=int, default=8, help="windows per config (CPU rows cost ~6 s each)")
    a = ap.parse_args()
    import numpy as np
    from PIL import Image
    os.makedirs(OUT, exist_ok=True)
    base_vram = vram_used()
    if base_vram and base_vram > 1000:
        print(f"WARNING: {base_vram} MB VRAM already in use (LLM running?) - GPU rows will be distorted.\n")
    win = os.path.join(OUT, "windows.npz")
    lo, hi = (int(v) for v in a.pages.split("-"))
    n = make_windows(a.pdf, lo, hi, win, a.windows)
    print(f"{n} real 512x512 windows from pages {lo}-{hi}\n")
    cfgs = [c for c in configs() if a.only in c["name"]]
    rows, outs = [], {}
    print(f"{'config':36s} {'status':8s} {'load s':>7s} {'s/win':>7s} {'VRAM MB':>8s} {'PSNR':>6s}  provider")
    for i, c in enumerate(cfgs):
        out_path = os.path.join(OUT, f"out_{i:02d}.npz")
        peak = {"v": base_vram or 0}
        stop = threading.Event()

        def sample():
            while not stop.is_set():
                v = vram_used()
                if v:
                    peak["v"] = max(peak["v"], v)
                time.sleep(0.5)
        th = threading.Thread(target=sample, daemon=True); th.start()
        status, res = "ok", {}
        try:
            p = subprocess.run([c["py"], os.path.abspath(__file__), "--child", json.dumps(c), win, out_path],
                               capture_output=True, text=True, timeout=TIMEOUT_S, cwd=ROOT)
            line = next((l for l in reversed(p.stdout.strip().splitlines()) if l.startswith("{")), None)
            res = json.loads(line) if line else {"error": (p.stderr or "no output").strip().splitlines()[-1][:160]}
            if "error" in res:
                status = "FAILED"
        except subprocess.TimeoutExpired:
            status, res = "TIMEOUT", {"error": f">{TIMEOUT_S}s"}
        finally:
            stop.set(); th.join(2)
        vram = (peak["v"] - (base_vram or 0)) if base_vram is not None else None
        ps = ""
        if status == "ok":
            outs[c["name"]] = np.load(out_path)["outs"]
        row = dict(c, status=status, vram_mb=vram, **res)
        rows.append(row)
        print(f"{c['name']:36s} {status:8s} {res.get('load_s', ''):>7} {res.get('per_window_s', ''):>7} "
              f"{vram if vram is not None else '?':>8} {'':>6}  {res.get('provider_used', res.get('error', ''))[:60]}",
              flush=True)
    # quality vs production (lama.onnx CPU) + sheets
    ref_name = next((k for k in outs if k.startswith("lama | CPU")), None)
    d = np.load(win)
    if ref_name:
        print(f"\nPSNR inside the mask vs production ({ref_name}), median over windows:")
        for k, o in outs.items():
            ps = [psnr_in_mask(x, r, m) for x, r, m in zip(o, outs[ref_name], d["masks"])]
            print(f"  {k:36s} {float(np.median(ps)):5.1f} dB")
            for r in rows:
                if r["name"] == k:
                    r["psnr_median_db"] = round(float(np.median(ps)), 1)
    best = {}
    for r in rows:
        if r["status"] == "ok":
            model = r["name"].split(" | ")[0]
            if model not in best or r["per_window_s"] < best[model]["per_window_s"]:
                best[model] = r
    cols = list(best)
    for k0 in range(0, len(d["imgs"]), 4):
        tiles = []
        for j in range(k0, min(k0 + 4, len(d["imgs"]))):
            row = [d["imgs"][j], np.stack([d["masks"][j]] * 3, -1)] + [outs[best[m]["name"]][j] for m in cols]
            tiles.append(np.concatenate(row, 1))
        Image.fromarray(np.concatenate(tiles, 0)).save(os.path.join(OUT, f"sheet_{k0 // 4}.jpg"), quality=88)
    json.dump(rows, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nsheets: {OUT}\\sheet_*.jpg  columns = original | mask | " + " | ".join(cols))
    print("fastest working config per model:")
    for m, r in best.items():
        print(f"  {m:18s} {r['name'].split(' | ')[1]:22s} {r['per_window_s']:.3f} s/win  VRAM +{r['vram_mb']} MB")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        main()
