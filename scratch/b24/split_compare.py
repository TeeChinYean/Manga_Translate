"""B24 quality guard: which boxes split differently with the new rules vs the old ones.
Run: python scratch/b24/split_compare.py "<pdf>" <first> <last>   -> prints changed boxes, saves crops"""
import os, sys, json, numpy as np, cv2, fitz
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pdf_translate"))
from core import extractor as X
pdf, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
doc = fitz.open(pdf); det = X._get_comic_detector(); outd = os.path.dirname(os.path.abspath(__file__))
tag = "v1" if "１" in pdf else "v5"
changed = 0
for pno in range(a, min(b, len(doc)) + 1):
    page = doc[pno - 1]; s = 850.0 / page.rect.height
    pix = page.get_pixmap(matrix=fitz.Matrix(s, s), alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()
    seg = []; boxes = X._detect_with_comic_detector(img, det, out_seg=seg)
    if not seg: continue
    for i, bx in enumerate(boxes):
        X.EXTENT_STRAY_SHARE, X.FURI_LEN_CHECK = 0.0, False
        old = X._split_box_by_seg(seg[0], bx)
        X.EXTENT_STRAY_SHARE, X.FURI_LEN_CHECK = 0.05, True
        new = X._split_box_by_seg(seg[0], bx)
        if len(old) != len(new):
            changed += 1
            x0, y0, x1, y1 = [int(v) for v in bx[:4]]
            vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)[max(0, y0 - 8):y1 + 8, max(0, x0 - 8):x1 + 8].copy()
            for q in new:
                cv2.rectangle(vis, (int(q[0]) - max(0, x0 - 8), int(q[1]) - max(0, y0 - 8)),
                              (int(q[2]) - max(0, x0 - 8), int(q[3]) - max(0, y0 - 8)), (0, 0, 255), 1)
            cv2.imwrite(os.path.join(outd, f"chg_{tag}_p{pno}_{i}.png"), vis)
            print(f"p{pno} box{i} {len(old)}->{len(new)}")
print("pages", a, b, "changed", changed)
