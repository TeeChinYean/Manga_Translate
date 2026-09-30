"""
Bilingual proofreading script as Excel (.xlsx), and the reverse: read the corrected
script back so the translation can be re-inserted into the same PDF.

Sheet "台本": one row per speech bubble. Only the 译文 column is meant to be edited.
The bubble bbox is kept in hidden columns so rows can be matched to the re-extracted
bubbles even if block ids shift. Sheet "meta" (hidden) records the source PDF hash
and the render parameters of the original run.
"""
import json
import math
import os
import time

from openpyxl import Workbook, load_workbook
from openpyxl.cell.cell import TYPE_STRING
from openpyxl.styles import Alignment, Font, PatternFill

SCRIPT_SHEET = "台本"
META_SHEET = "meta"
HEADERS = ["页码", "气泡", "原文 (OCR)", "译文 (可修改)", "x0", "y0", "x1", "y1"]
COL_TRANSLATION = 4
MATCH_MIN_IOU = 0.5
HEADER_MAX_CHANGED = 2   # header cells that may differ before the sheet is rejected
MAX_XLSX_BYTES = 20 * 1024 * 1024


class CorrectionsError(ValueError):
    """The uploaded Excel is not a proofreading script produced by this tool."""


def _text_cell(ws, row, col, value):
    # Always store as text: a translation starting with "=" must not become a formula
    cell = ws.cell(row=row, column=col)
    cell.value = "" if value is None else str(value)
    cell.data_type = TYPE_STRING
    return cell


class LocalDocumentSkill:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_bilingual_xlsx(self, task_id: str, pages_data: list, original_filename: str,
                                meta: dict | None = None) -> str:
        """
        pages_data: [{"page_num": 1, "blocks": [{"id": 3, "raw": "...", "translated": "...",
                      "bbox": [x0, y0, x1, y1]}, ...]}, ...]
        meta: extra key/value pairs for the hidden meta sheet (pdf_md5, render params, ...).
        """
        stem = os.path.splitext(original_filename)[0]
        path = os.path.join(self.output_dir, f"script_{task_id[:8]}_{stem}.xlsx")

        wb = Workbook()
        ws = wb.active
        ws.title = SCRIPT_SHEET
        ws.append(HEADERS)
        for c in ws[1]:
            c.font = Font(bold=True)
            c.fill = PatternFill("solid", fgColor="E6F7FF")
        edit_fill = PatternFill("solid", fgColor="FFF7E0")
        wrap = Alignment(wrap_text=True, vertical="top")

        row = 2
        for page in pages_data:
            for blk in page.get("blocks", []):
                # A box with no OCR text is still exported when it has a translation (typed by hand in the box editor)
                if not str(blk.get("raw", "")).strip() and not str(blk.get("translated", "")).strip():
                    continue
                ws.cell(row=row, column=1, value=int(page["page_num"]))
                ws.cell(row=row, column=2, value=int(blk["id"]))
                _text_cell(ws, row, 3, blk.get("raw", "")).alignment = wrap
                t = _text_cell(ws, row, COL_TRANSLATION, blk.get("translated", ""))
                t.alignment = wrap
                t.fill = edit_fill
                for i, v in enumerate(blk.get("bbox") or [0, 0, 0, 0]):
                    ws.cell(row=row, column=5 + i, value=round(float(v), 2))
                row += 1

        ws.freeze_panes = "A2"
        for col, width in zip("ABCD", (7, 7, 45, 45)):
            ws.column_dimensions[col].width = width
        for col in "EFGH":
            ws.column_dimensions[col].hidden = True

        ms = wb.create_sheet(META_SHEET)
        ms.sheet_state = "hidden"
        info = {"source_file": original_filename, "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
        info.update(meta or {})
        for k, v in info.items():
            ms.append([k])
            _text_cell(ms, ms.max_row, 2, v)

        wb.save(path)
        return path


def read_corrections(xlsx_path) -> tuple[dict, dict]:
    """
    `xlsx_path`: a path or a binary file-like object (e.g. io.BytesIO of the upload).
    Returns (meta, corrections) where corrections = {page_num: [{"id", "bbox", "text"}, ...]}.
    Raises CorrectionsError for anything that is not a script from generate_bilingual_xlsx.
    """
    try:
        wb = load_workbook(xlsx_path, read_only=True, data_only=True)
    except Exception as e:
        raise CorrectionsError(f"无法读取 Excel 文件: {e}")
    try:
        if SCRIPT_SHEET not in wb.sheetnames or META_SHEET not in wb.sheetnames:
            raise CorrectionsError("这不是本系统导出的校对台本（缺少「台本」或 meta 工作表）")
        meta = {}
        for r in wb[META_SHEET].iter_rows(values_only=True):
            if r and r[0]:
                meta[str(r[0])] = "" if len(r) < 2 or r[1] is None else str(r[1])

        rows = wb[SCRIPT_SHEET].iter_rows(values_only=True)
        header = next(rows, None)
        # Columns are read by position, so a stray edit in one header cell (e.g. a translation
        # pasted into A1) is harmless; only reject when the layout itself looks different.
        header = list(header or [])[:len(HEADERS)]
        header += [None] * (len(HEADERS) - len(header))
        bad = [f"{chr(65 + i)}1" for i, (h, want) in enumerate(zip(header, HEADERS))
               if str(h or "").strip() != want]
        if len(bad) > HEADER_MAX_CHANGED:
            raise CorrectionsError(f"「台本」工作表的表头被修改了（{', '.join(bad)}），请使用原始导出的文件")
        corrections = {}
        for r in rows:
            if not r or r[0] is None or r[1] is None:
                continue
            try:
                page, bid = int(r[0]), int(r[1])
                bbox = [float(v) for v in r[4:8]] if len(r) >= 8 and None not in r[4:8] else None
            except (TypeError, ValueError):
                continue
            text = "" if r[3] is None else str(r[3])
            raw = "" if r[2] is None else str(r[2])
            corrections.setdefault(page, []).append({"id": bid, "bbox": bbox, "text": text, "raw": raw})
        return meta, corrections
    finally:
        wb.close()


MAX_EDITOR_ROWS = 5000
MAX_EDITOR_TEXT = 2000
MIN_FONT_PT, MAX_FONT_PT = 4.0, 150.0     # box editor: lettering size chosen by the user, in PDF points (0 = automatic)


def clean_font_size(value) -> float:
    """Font size (pt) from the box editor: 0 = automatic, otherwise clamped to MIN_FONT_PT..MAX_FONT_PT."""
    try:
        v = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v) or v <= 0:
        return 0.0
    return round(min(max(v, MIN_FONT_PT), MAX_FONT_PT), 1)


DIRECTIONS = ("horizontal", "vertical", "vertical_rtl", "auto")   # lettering direction; keep in sync with core.renderer.TEXT_DIRECTIONS


def parse_corrections_json(text: str) -> dict:
    """
    Rows from the box editor (same shape as read_corrections): JSON
    {"pages": {"3": [{"id": 1, "bbox": [x0, y0, x1, y1], "text": "译文", "raw": "原文"}, ...]}}.
    bbox is in PDF points, origin top-left. A page with an empty list is kept (rendered without
    any bubble). Returns {page_num: [{"id", "bbox", "text", "raw"}]}; raises CorrectionsError.
    """
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        raise CorrectionsError("框编辑数据不是合法的 JSON")
    pages = data.get("pages") if isinstance(data, dict) else None
    if not isinstance(pages, dict) or not pages:
        raise CorrectionsError("框编辑数据里没有页面")
    out, total = {}, 0
    for key, rows in pages.items():
        try:
            page = int(key)
        except (TypeError, ValueError):
            raise CorrectionsError(f"页码无效: {key}")
        if page < 1 or not isinstance(rows, list):
            raise CorrectionsError(f"第 {key} 页的数据无效")
        clean, seen = [], set()
        for r in rows:
            try:
                bid = int(r["id"])
                bbox = [float(v) for v in r["bbox"]]
            except (KeyError, TypeError, ValueError):
                raise CorrectionsError(f"第 {page} 页有一个框的数据不完整")
            if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox) \
                    or bbox[2] - bbox[0] < 1 or bbox[3] - bbox[1] < 1:
                raise CorrectionsError(f"第 {page} 页的框 #{bid} 坐标无效")
            if bid in seen:
                raise CorrectionsError(f"第 {page} 页的框编号 #{bid} 重复")
            seen.add(bid)
            txt = "" if r.get("text") is None else str(r["text"])
            if len(txt) > MAX_EDITOR_TEXT:
                raise CorrectionsError(f"第 {page} 页的框 #{bid} 译文过长")
            raw = "" if r.get("raw") is None else str(r["raw"])[:MAX_EDITOR_TEXT]
            direction = r.get("direction") if r.get("direction") in DIRECTIONS else ""
            clean.append({"id": bid, "bbox": bbox, "text": txt, "raw": raw, "edited": bool(r.get("edited")),
                          "direction": direction, "font_size": clean_font_size(r.get("font_size"))})
        total += len(clean)
        out[page] = clean
    if total > MAX_EDITOR_ROWS:
        raise CorrectionsError("框数量过多")
    return out


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_corrections(blocks: list, rows: list, unmatched: list | None = None) -> dict:
    """
    Map re-extracted blocks of one page to corrected rows: best bbox IoU (>= MATCH_MIN_IOU),
    same block id breaks ties. Each row is used once. Returns {block_id: text}.
    Blocks without a match get nothing, so the renderer keeps their original text.
    `unmatched`: if given, rows with a non-empty translation that matched no block are appended.
    """
    pairs = []
    for bi, blk in enumerate(blocks):
        for ri, row in enumerate(rows):
            if row["bbox"] is not None:
                score = _iou(blk["bbox"], row["bbox"])
                if score < MATCH_MIN_IOU:
                    continue
            elif row["id"] == blk["id"]:
                score = MATCH_MIN_IOU  # no bbox in the sheet: fall back to the id
            else:
                continue
            pairs.append((score, row["id"] == blk["id"], bi, ri))
    pairs.sort(reverse=True)
    used_b, used_r, out = set(), set(), {}
    for _, _, bi, ri in pairs:
        if bi in used_b or ri in used_r:
            continue
        used_b.add(bi)
        used_r.add(ri)
        out[blocks[bi]["id"]] = rows[ri]["text"]
    if unmatched is not None:
        unmatched.extend(r for ri, r in enumerate(rows) if ri not in used_r and str(r["text"]).strip())
    return out


def blocks_from_corrections(rows: list) -> list:
    """Bubbles of one page rebuilt from the Excel rows (re-insert without OCR). Rows without
    a bbox cannot be placed and are left out (reported as unmatched by match_corrections)."""
    blocks = []
    for r in rows:
        if not r.get("bbox"):
            continue
        x0, y0, x1, y1 = r["bbox"]
        raw = r.get("raw", "")
        blocks.append({"id": r["id"], "text": raw, "cleaned_text": raw, "bbox": [x0, y0, x1, y1],
                       "lines_bboxes": [[x0, y0, x1, y1]], "font_size": 12.0, "font_name": "Helvetica",
                       "color": (0.0, 0.0, 0.0), "width": x1 - x0, "height": y1 - y0,
                       "center_x": (x0 + x1) / 2.0, "center_y": (y0 + y1) / 2.0,
                       "ocr_engine": "Excel 台本", "user_edited": bool(r.get("edited"))})
        if r.get("direction") in DIRECTIONS:
            blocks[-1]["direction"] = r["direction"]     # per-box lettering direction (box editor)
        if clean_font_size(r.get("font_size")):
            blocks[-1]["font_size_pt"] = clean_font_size(r.get("font_size"))   # per-box lettering size (box editor)
    return blocks


def build_unmatched_warning(unmatched_by_page: dict) -> str:
    """{page_num: n_rows} -> user message, or "" when every corrected row found its bubble."""
    pages = sorted(p for p, n in unmatched_by_page.items() if n)
    if not pages:
        return ""
    total = sum(unmatched_by_page[p] for p in pages)
    return (f"{total} 行校对译文未匹配到气泡（该气泡保留原文）: "
            + ", ".join(f"第 {p} 页 {unmatched_by_page[p]} 行" for p in pages))
