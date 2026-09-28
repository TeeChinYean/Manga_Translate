# Manual check for BUG.md B7: compare MangaOCR vs PaddleOCR on the sample manga.
# Run on Windows from repo root:  pdf_translate\.venv\Scripts\python scratch\compare_ja_ocr.py
import os, sys, logging
logging.disable(logging.CRITICAL)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pdf_translate"))
import fitz, numpy as np
from PIL import Image
from core.extractor import _get_comic_detector, _detect_with_comic_detector, _get_paddle_ocr, _get_manga_ocr

pdf = os.path.join(ROOT, "test_ori_clip.pdf")
doc = fitz.open(pdf)
det, paddle, mocr = _get_comic_detector(), _get_paddle_ocr(), _get_manga_ocr()
print("MangaOCR loaded:", mocr is not None)
for pn in range(len(doc)):
    page = doc[pn]
    s = 850.0 / page.rect.height
    pix = page.get_pixmap(matrix=fitz.Matrix(s, s), alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    for (x0, y0, x1, y1) in _detect_with_comic_detector(img, det):
        crop = img[max(0, y0 - 8):y1 + 8, max(0, x0 - 8):x1 + 8]
        p_txt = "".join(l[1] for l in (paddle(crop)[0] or [])) if paddle else "-"
        m_txt = mocr(Image.fromarray(crop)) if mocr else "-"
        print(f"page {pn + 1} | Paddle: {p_txt:<20} | MangaOCR: {m_txt}")
