# Can LaMa get faster on CPU?  lama.onnx has a FIXED 512x512 input, so a smaller inference
# size (384) is impossible without re-exporting the model. What IS possible: pack several
# small text regions (from all pages, serial mode has them all) into one 512x512 canvas and
# inpaint them with ONE LaMa call instead of one call each.
#
# This script compares, on real pages:
#   baseline = production (one LaMa call per region group, 512 window around it)
#   packed   = regions shelf-packed into shared 512 canvases
# and reports LaMa calls, time, and PSNR of packed vs baseline inside the masks.
# Images (original | baseline | packed) go to scratch\lama_pack_compare\.
#
# Run from repo root (LLM can stay running, this is CPU only):
#   python scratch\lama_pack_test.py                    (第5巻.pdf, pages 1-10)
#   python scratch\lama_pack_test.py x.pdf --pages 1-20 --gap 24
import argparse, os, sys, time, logging
logging.disable(logging.WARNING)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pdf_translate"))
import numpy as np, cv2, fitz
from PIL import Image
from core import renderer as R
from core.extractor import _get_comic_detector, _detect_with_comic_detector

OUT = os.path.join(ROOT, "scratch", "lama_pack_compare")
S = R.LAMA_SIZE


def parse_pages(spec, n):
    pages = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            pages.update(range(int(a), int(b) + 1))
        elif part.strip():
            pages.add(int(part))
    return [p for p in sorted(pages) if 1 <= p <= n]


def page_masks(pdf, pages):
    """(page_num, rgb image, binary LaMa mask) exactly as production builds them."""
    doc, det, rend, out = fitz.open(pdf), _get_comic_detector(), R.PDFLayoutRenderer(), []
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
        if prep["lama_mask"] is not None:
            _, m = cv2.threshold(prep["lama_mask"], 10, 255, cv2.THRESH_BINARY)
            out.append((pn, np.array(img), m))
    return out


def baseline(pages_data):
    """Production path: R._lama_inpaint per page (one call per region group)."""
    res, calls, t0 = {}, 0, time.perf_counter()
    for pn, img, m in pages_data:
        calls += len(R._lama_regions(m, img.shape[1], img.shape[0]))
        res[pn] = np.array(R._lama_inpaint(Image.fromarray(img), Image.fromarray(m)))
    return res, calls, time.perf_counter() - t0


def region_tiles(pages_data, pad):
    """Small regions (fit in a 512 canvas with context) become tiles; big ones stay alone."""
    tiles, big = [], []
    for pn, img, m in pages_data:
        h, w = m.shape
        n, _, stats, _ = cv2.connectedComponentsWithStats(
            cv2.dilate((m > 0).astype(np.uint8), np.ones((R.LAMA_MERGE_PX, R.LAMA_MERGE_PX), np.uint8)), 8)
        for i in range(1, n):
            x, y, bw, bh = (int(v) for v in stats[i, :4])
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
            item = (pn, x0, y0, x1, y1)
            (tiles if max(x1 - x0, y1 - y0) <= S else big).append(item)
    return tiles, big


def shelf_pack(tiles, gap):
    """Greedy shelf packing into SxS canvases. Returns list of canvases: [(tile, cx, cy), ...]."""
    canvases, cur, x, y, shelf_h = [], [], 0, 0, 0
    for t in sorted(tiles, key=lambda t: -(t[4] - t[2])):
        tw, th = t[3] - t[1], t[4] - t[2]
        if x + tw > S:
            x, y, shelf_h = 0, y + shelf_h + gap, 0
        if y + th > S:
            canvases.append(cur)
            cur, x, y, shelf_h = [], 0, 0, 0
        cur.append((t, x, y))
        x += tw + gap
        shelf_h = max(shelf_h, th)
    if cur:
        canvases.append(cur)
    return canvases


def packed(pages_data, pad, gap):
    by_page = {pn: (img, m) for pn, img, m in pages_data}
    res = {pn: img.copy() for pn, img, _ in pages_data}
    tiles, big = region_tiles(pages_data, pad)
    canvases = shelf_pack(tiles, gap)
    session = R._get_lama_session()
    t0 = time.perf_counter()
    for items in canvases:
        canvas = np.full((S, S, 3), 255, np.uint8)
        cmask = np.zeros((S, S), np.uint8)
        for (pn, x0, y0, x1, y1), cx, cy in items:
            img, m = by_page[pn]
            canvas[cy:cy + y1 - y0, cx:cx + x1 - x0] = img[y0:y1, x0:x1]
            cmask[cy:cy + y1 - y0, cx:cx + x1 - x0] = m[y0:y1, x0:x1]
        out = R._lama_inpaint_crop(session, canvas, cmask)
        for (pn, x0, y0, x1, y1), cx, cy in items:
            sel = by_page[pn][1][y0:y1, x0:x1] > 0
            res[pn][y0:y1, x0:x1][sel] = out[cy:cy + y1 - y0, cx:cx + x1 - x0][sel]
    calls = len(canvases)
    for (pn, x0, y0, x1, y1) in big:  # large regions: production path on their own
        img, m = by_page[pn]
        sub = np.zeros_like(m)
        sub[y0:y1, x0:x1] = m[y0:y1, x0:x1]
        calls += len(R._lama_regions(sub, m.shape[1], m.shape[0]))
        out = np.array(R._lama_inpaint(Image.fromarray(res[pn]), Image.fromarray(sub)))
        res[pn][sub > 0] = out[sub > 0]
    return res, calls, time.perf_counter() - t0, len(tiles), len(big)


def psnr(a, b, m):
    sel = m > 0
    if not sel.any():
        return 99.0
    mse = float(((a[sel].astype(np.float32) - b[sel].astype(np.float32)) ** 2).mean())
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?", default=os.path.join(ROOT, "第5巻.pdf"))
    ap.add_argument("--pages", default="1-10")
    ap.add_argument("--pad", type=int, default=R.LAMA_CONTEXT_PAD, help="context px around each tile")
    ap.add_argument("--gap", type=int, default=16, help="px between tiles on a canvas")
    a = ap.parse_args()

    print("lama.onnx input:", [(i.name, i.shape) for i in R._get_lama_session().get_inputs()],
          "(fixed 512 -> a 384 inference size is not possible)")
    data = page_masks(a.pdf, parse_pages(a.pages, len(fitz.open(a.pdf))))
    print(f"{len(data)} page(s) with textured regions")
    if not data:
        sys.exit("nothing to inpaint on these pages")

    print("running baseline (production) ...", flush=True)
    base, b_calls, b_time = baseline(data)
    print("running packed ...", flush=True)
    pack, p_calls, p_time, n_tiles, n_big = packed(data, a.pad, a.gap)

    os.makedirs(OUT, exist_ok=True)
    print(f"\npage  PSNR packed vs baseline (dB, in mask)")
    worst = 99.0
    for pn, img, m in data:
        v = psnr(base[pn], pack[pn], m)
        worst = min(worst, v)
        print(f"{pn:>4}  {v:6.1f}")
        ys, xs = np.nonzero(m)
        y0, y1 = max(0, ys.min() - 40), min(m.shape[0], ys.max() + 40)
        x0, x1 = max(0, xs.min() - 40), min(m.shape[1], xs.max() + 40)
        side = np.concatenate([img[y0:y1, x0:x1], base[pn][y0:y1, x0:x1], pack[pn][y0:y1, x0:x1]], axis=1)
        Image.fromarray(side).save(os.path.join(OUT, f"page_{pn:03d}.png"))
    print(f"\nbaseline: {b_calls} LaMa calls, {b_time:.1f}s")
    print(f"packed  : {p_calls} LaMa calls ({n_tiles} small tiles, {n_big} large regions), {p_time:.1f}s")
    print(f"speed-up x{b_time / max(p_time, 1e-6):.2f} | worst page PSNR {worst:.1f} dB "
          f"(>=30 visually identical, 25-30 look at images, <25 visible difference)")
    print(f"compare images: {OUT}  (original | baseline | packed)")
