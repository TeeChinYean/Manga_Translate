"""B24 debug: run CTD on a page, draw raw / split boxes + seg mask.
Run: python scratch/b24/b24_debug.py "<pdf>" 26"""
import os, sys
import numpy as np, cv2, fitz
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pdf_translate"))
from core import extractor as X
pdf, pno = sys.argv[1], int(sys.argv[2])
doc = fitz.open(pdf); page = doc[pno - 1]
s = 850.0 / page.rect.height
pix = page.get_pixmap(matrix=fitz.Matrix(s, s), alpha=False)
img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()
seg = []
boxes = X._detect_with_comic_detector(img, X._get_comic_detector(), out_seg=seg)
vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
segm = seg[0]
ov = vis.copy(); ov[segm > 0] = (0, 0, 255); vis = cv2.addWeighted(vis, 0.6, ov, 0.4, 0)
for i, b in enumerate(boxes):
    cv2.rectangle(vis, tuple(map(int, b[:2])), tuple(map(int, b[2:4])), (255, 0, 0), 2)
    cv2.putText(vis, str(i), (int(b[0]), int(b[1]) + 12), 0, 0.5, (255, 0, 0), 2)
    sp = X._split_box_by_seg(segm, b)
    print(i, [int(v) for v in b[:4]], "->", len(sp), [[int(v) for v in q[:4]] for q in sp])
    for q in sp if len(sp) > 1 else []:
        cv2.rectangle(vis, tuple(map(int, q[:2])), tuple(map(int, q[2:4])), (0, 200, 0), 1)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"p{pno}.png")
cv2.imwrite(out, vis); print(out, segm.shape, segm.dtype, segm.max())
