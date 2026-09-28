# Compare MangaOCR one-crop-at-a-time vs batched on real crops (same Python as start_web_app.bat).
#   python scratch\mangaocr_batch_test.py                (第5巻.pdf, pages 1-5)
#   python scratch\mangaocr_batch_test.py x.pdf --pages 1-10 --batches 4,8,16
import argparse, os, sys, time, logging
logging.disable(logging.WARNING)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pdf_translate"))
import numpy as np, fitz
from PIL import Image
from core.extractor import (_get_comic_detector, _detect_with_comic_detector, _split_box_by_seg,
                            _get_manga_ocr, _manga_ocr_batch)


def crops(pdf, pages):
    doc, det, out = fitz.open(pdf), _get_comic_detector(), []
    for pn in pages:
        page = doc[pn - 1]
        s = 850.0 / page.rect.height
        pix = page.get_pixmap(matrix=fitz.Matrix(s, s), alpha=False)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
        seg = []
        for b in _detect_with_comic_detector(img, det, out_seg=seg):
            for x0, y0, x1, y1 in _split_box_by_seg(seg[0], b):
                c = img[max(0, y0 - 8):y1 + 8, max(0, x0 - 8):x1 + 8]
                if c.size:
                    out.append(Image.fromarray(c))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?", default=os.path.join(ROOT, "第5巻.pdf"))
    ap.add_argument("--pages", default="1-5")
    ap.add_argument("--batches", default="4,8,16")
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.pages.split("-"))
    ims = crops(a.pdf, range(lo, hi + 1))
    mocr = _get_manga_ocr()
    if mocr is None:
        sys.exit("MangaOCR not available in this Python")
    print(f"{len(ims)} crops, device {mocr.model.device}")
    _manga_ocr_batch(mocr, ims[:2], batch_size=2)  # warm-up
    t = time.perf_counter()
    base = _manga_ocr_batch(mocr, ims, batch_size=1)
    t1 = time.perf_counter() - t
    print(f"batch  1 : {t1:6.1f}s  ({t1 / len(ims):.2f}s/crop)")
    for bs in (int(v) for v in a.batches.split(",")):
        t = time.perf_counter()
        got = _manga_ocr_batch(mocr, ims, batch_size=bs)
        tb = time.perf_counter() - t
        same = sum(g == b for g, b in zip(got, base))
        print(f"batch {bs:2d} : {tb:6.1f}s  ({tb / len(ims):.2f}s/crop)  x{t1 / tb:.2f}  identical text {same}/{len(ims)}")
        for g, b in zip(got, base):
            if g != b:
                print(f"    differs: {b!r} -> {g!r}")
