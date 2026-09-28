# Why do white edges / blotches stay around text drawn over artwork?
# Dumps, per page: original | mask overlay (red=LaMa, blue=Telea) | inpainted WITHOUT new text | final.
# Output: scratch\mask_debug\<pdf-stem>_p<N>.jpg  (send me the images, or I stage them)
#
# Run from repo root (LLM not needed: dummy text is drawn):
#   python scratch\mask_debug.py "pdf_translate\data\uploads\preview_a6adbf5c16_図書館の大魔術師 第３巻.pdf" --pages 150,153
import argparse, os, sys, logging
logging.disable(logging.WARNING)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pdf_translate"))
import numpy as np, fitz
from PIL import Image
from core import renderer as R
from core.extractor import PDFLayoutExtractor

OUT = os.path.join(ROOT, "scratch", "mask_debug")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--pages", default="1")
    ap.add_argument("--dump-lama", action="store_true",
                    help="also save every LaMa window: input | mask | 512 output | pasted result")
    a = ap.parse_args()
    if a.dump_lama:
        _orig = R._lama_inpaint_crop
        counter = {"n": 0}

        def _spy(session, crop_rgb, crop_mask):
            out = _orig(session, crop_rgb, crop_mask)
            counter["n"] += 1
            h, w = crop_rgb.shape[:2]
            m3 = np.stack([crop_mask] * 3, -1)
            pasted = np.where(m3 > 0, out, crop_rgb)
            tile = np.concatenate([crop_rgb, m3, out, pasted], axis=1)
            Image.fromarray(tile).save(os.path.join(OUT, f"lama_{counter['n']:02d}_{w}x{h}.jpg"), quality=90)
            print(f"   lama window {counter['n']}: {w}x{h}, mask {int((crop_mask > 0).mean() * 100)}%", flush=True)
            return out

        R._lama_inpaint_crop = _spy
    os.makedirs(OUT, exist_ok=True)
    pages = [int(x) for x in a.pages.split(",") if x.strip()]
    ex = PDFLayoutExtractor(a.pdf)
    rd = R.PDFLayoutRenderer(original_pdf_path=a.pdf)
    doc = fitz.open(a.pdf)
    for p in pages:
        data = ex._extract_single_page(a.pdf, p, "Japanese")
        tmap = {(p, b["id"]): "测试文字" for b in data["blocks"]}
        img, scale = rd._rasterize(doc[p - 1])
        prep = rd._build_masks(img, scale, data, tmap)
        base = np.array(img)
        ov = base.copy()
        if prep["lama_mask"] is not None:
            ov[prep["lama_mask"] > 0] = (0.4 * ov[prep["lama_mask"] > 0] + [153, 0, 0]).astype(np.uint8)
        if prep["telea_mask"] is not None:
            ov[prep["telea_mask"] > 0] = (0.4 * ov[prep["telea_mask"] > 0] + [0, 0, 153]).astype(np.uint8)
        healed = img
        if prep["telea_mask"] is not None:
            healed = R._fast_telea_inpaint(healed, Image.fromarray(prep["telea_mask"]))
        if prep["lama_mask"] is not None:
            healed = R._lama_inpaint(healed, Image.fromarray(prep["lama_mask"]))  # uses R._lama_inpaint_crop
        clean = np.array(healed).copy()
        final = np.array(rd._draw_translations(healed.copy(), prep["blocks"]))
        strip = np.concatenate([base, ov, clean, final], axis=1)
        name = f"{os.path.splitext(os.path.basename(a.pdf))[0][-12:]}_p{p}.jpg"
        Image.fromarray(strip).save(os.path.join(OUT, name), quality=88)
        nl = int(np.count_nonzero(prep["lama_mask"])) if prep["lama_mask"] is not None else 0
        nt = int(np.count_nonzero(prep["telea_mask"])) if prep["telea_mask"] is not None else 0
        print(f"page {p}: blocks={len(prep['blocks'])} lama_px={nl} telea_px={nt} -> {os.path.join(OUT, name)}", flush=True)


if __name__ == "__main__":
    main()
