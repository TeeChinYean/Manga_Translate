import os, sys, numpy as np, cv2, fitz
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pdf_translate"))
from core import extractor as X
pdf, pno = sys.argv[1], int(sys.argv[2]); ids = [int(v) for v in sys.argv[3:]]
page = fitz.open(pdf)[pno - 1]; s = 850.0 / page.rect.height
pix = page.get_pixmap(matrix=fitz.Matrix(s, s), alpha=False)
img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()
seg = []; boxes = X._detect_with_comic_detector(img, X._get_comic_detector(), out_seg=seg); seg = seg[0]
for i in ids:
    x0, y0, x1, y1 = [int(v) for v in boxes[i][:4]]
    r = seg[y0:y1, x0:x1] >= X.SPLIT_SEG_THRESH
    runs = X._runs(r.any(axis=0))
    print("box", i, (x0, y0, x1, y1), "runs(x):", runs)
    for a, b in runs:
        ys = np.nonzero(r[:, a:b].any(axis=1))[0]; print("   col", a, b, "y", ys.min(), ys.max(), "px", int(r[:, a:b].sum()))
    rows = X._runs(r.any(axis=1)); print("   runs(y):", rows)
    cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"seg_{pno}_{i}.png"),
                np.hstack([cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY), (r * 255).astype(np.uint8)]))
    for a, b in runs:
        cnt = r[:, a:b].sum(axis=1)
        print("   col", a, b, "rowcounts:", "".join("." if c == 0 else (str(min(9, c))) for c in cnt))
