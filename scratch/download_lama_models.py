# Download the newer LaMa ONNX models for the GPU / quality test (~415 MB total).
#   python scratch\download_lama_models.py
# Saves to pdf_translate\data\models\onnx\ (existing files are skipped). Current lama.onnx is untouched.
import os, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "pdf_translate", "data", "models", "onnx")
FILES = [
    # Carve re-export: opset 17, FourierUnitJIT, "identical to the original", TensorRT-able
    ("https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx", "lama_fp32.onnx"),
    # Manga/anime-trained LaMa (AnimeMangaInpainting, as used by manga-image-translator), Apache-2.0
    ("https://huggingface.co/mayocream/lama-manga-onnx/resolve/main/lama-manga.onnx", "lama-manga.onnx"),
]


def fetch(url, path):
    tmp = path + ".part"
    t0, last = time.time(), 0.0
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if time.time() - last > 1:
                last = time.time()
                pct = f"{done * 100 / total:5.1f}%" if total else ""
                print(f"\r  {os.path.basename(path)}  {done / 2**20:7.1f} MB {pct}  "
                      f"{done / 2**20 / max(1e-6, time.time() - t0):5.1f} MB/s", end="", flush=True)
    os.replace(tmp, path)
    print(f"\r  {os.path.basename(path)}  {os.path.getsize(path) / 2**20:.1f} MB  done in {time.time() - t0:.0f}s        ")


os.makedirs(DEST, exist_ok=True)
for url, name in FILES:
    path = os.path.join(DEST, name)
    if os.path.exists(path) and os.path.getsize(path) > 100 * 2**20:
        print(f"  {name} already there ({os.path.getsize(path) / 2**20:.0f} MB), skipped")
        continue
    try:
        fetch(url, path)
    except Exception as e:
        print(f"\n  FAILED {name}: {e}")
        sys.exit(1)
print(f"\nall files in {DEST}")
