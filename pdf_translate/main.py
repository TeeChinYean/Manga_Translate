#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ ANTIGRAVITY - PDF Layout Preservation Translation Engine Backend
Technology Stack: FastAPI, Asyncio Queue, Server-Sent Events (SSE) Stream, PyMuPDF.
Features: Decoupled from RAG terminology, simplified direct pipeline.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import io
import os
# torch.cuda.is_available() via NVML: no CUDA driver init in the web server process (B29)
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
import sys
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import uuid
import json
import asyncio
import logging
import multiprocessing
import hashlib
import zipfile
import shutil
import tempfile
import fitz

# Dynamic CPU core detection and multi-threading optimization
num_cores = multiprocessing.cpu_count()
# Limit threads to a balanced size (typically 4 threads is the sweet spot for PyTorch CPU inference to prevent thrashing)
recommended_threads = max(2, min(4, num_cores // 2))

os.environ["OMP_NUM_THREADS"] = str(recommended_threads)
os.environ["MKL_NUM_THREADS"] = str(recommended_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(recommended_threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(recommended_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(recommended_threads)

# Configure PyTorch and OpenCV to utilize a balanced number of CPU cores
try:
    import torch
    torch.set_num_threads(recommended_threads)
    logging.info(f"⚡ [CPU Optimizer] PyTorch thread pool size set to {recommended_threads} cores (preventing thread thrashing).")
except ImportError:
    pass

try:
    import cv2
    cv2.setNumThreads(num_cores)
    logging.info(f"⚡ [CPU Optimizer] OpenCV thread pool size set to {num_cores} cores.")
except ImportError:
    pass

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, Response
from typing import List, Optional
from pydantic import BaseModel
from starlette.background import BackgroundTask

import fitz

# Import core pipeline components
from core.extractor import PDFLayoutExtractor
from core.engine import HighPerformanceTranslationEngine
from core.renderer import PDFLayoutRenderer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="⚡ ANTIGRAVITY - PDF Layout Preservation Translation Engine",
    description="Accelerated PDF translation pipeline with layout preservation and speculative decoding.",
    version="1.0.0"
)

# Directories setup — resolved relative to this script's location (cross-platform)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
STATIC_DIR = os.path.join(BASE_DIR, "static")
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")  # page cache of the C# entry (core/pipeline.py); swept on startup

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Global memory databases for async queues
translation_queue = asyncio.Queue()
status_db = {}  # {task_id: {"percent": int, "stage": str, "status": str, "download_url": str, "metrics": dict}}
cancelled_tasks = set()

# Initialize system engine
translation_engine = None

def parse_page_range(range_str: str, max_pages: int) -> set:
    """
    Parses a page range string (e.g. "1-10, 15, 20-30") into a set of 1-based page numbers.
    """
    if not range_str or not range_str.strip():
        return set(range(1, max_pages + 1))
    
    pages = set()
    clean_str = range_str.replace("，", ",").replace("－", "-").replace("—", "-")
    parts = clean_str.split(",")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                start, end = part.split("-")
                start = int(start.strip())
                end = int(end.strip())
                for p in range(start, end + 1):
                    if 1 <= p <= max_pages:
                        pages.add(p)
            except ValueError:
                pass
        else:
            try:
                p = int(part)
                if 1 <= p <= max_pages:
                    pages.add(p)
            except ValueError:
                pass
                
    return pages if pages else set(range(1, max_pages + 1))


PIPELINE_MODES = ("auto", "stream", "overlap", "serial")
# 'auto': serial for short jobs, overlap for long ones. Measured 2026-09-28 on 第5巻:
#   10 pages: serial 95.5s (translation ~10s, nothing to hide)
#   50 pages: overlap 472.1s vs serial 566.5s (overlap hides ~145s of LLM time behind extraction)
AUTO_OVERLAP_MIN_PAGES = max(1, int(os.getenv("AUTO_OVERLAP_MIN_PAGES", "20")))
# 'auto' with GPU LaMa: overlap (extract || translate, render after translation). AUTO_GPU_MODE=serial = old
AUTO_GPU_MODE = os.getenv("AUTO_GPU_MODE", "overlap").strip().lower()
if AUTO_GPU_MODE not in ("overlap", "serial", "stream"):
    AUTO_GPU_MODE = "overlap"


def resolve_pipeline_mode(mode: str, n_pages: int) -> str:
    """Map 'auto' (or an unknown value) to a concrete mode for this job size."""
    if mode not in PIPELINE_MODES or mode == "auto":
        try:
            from core.renderer import lama_uses_gpu
            if lama_uses_gpu():
                # GPU LaMa must not share the 4 GB card with the LLM (B28), but extraction runs
                # on the CPU, so extract + translate may overlap: overlap mode then holds
                # rendering until translation is done (user measured overlap faster, 2026-09-29).
                return AUTO_GPU_MODE
        except Exception:
            pass
        return "overlap" if n_pages >= AUTO_OVERLAP_MIN_PAGES else "serial"
    return mode
# Measured 2026-09-28 (第5巻 p1-10, RTX 3050 4GB + 12 threads): serial 109.2s vs stream 129.7s,
# and serial translates the whole range with one large-context LLM call.
DEFAULT_PIPELINE_MODE = os.getenv("PIPELINE_MODE", "auto")
DEFAULT_CONTEXT_CHUNK = 36  # Japanese lines per LLM call in 'serial' mode
RENDER_CONCURRENCY = max(1, int(os.getenv("RENDER_CONCURRENCY", "3")))  # pages rendered at once (all modes)
# stream/overlap: translate once this many lines are buffered (0 = old per-page behaviour)
DEFAULT_STREAM_BATCH_LINES = max(0, int(os.getenv("TRANSLATE_BATCH_LINES", str(DEFAULT_CONTEXT_CHUNK))))


def clear_dir_contents(path: str) -> int:
    """Delete every file and sub-directory in `path` except .git* placeholders. Returns how many were removed."""
    try:
        names = [n for n in os.listdir(path) if not n.startswith(".git")]
    except OSError:
        return 0
    removed = 0
    for name in names:
        full = os.path.join(path, name)
        try:
            if os.path.isdir(full):
                shutil.rmtree(full)
            else:
                os.unlink(full)
            removed += 1
        except OSError as e:
            logger.warning(f"Failed to remove {full}: {e}")
    return removed


def clear_leftovers() -> None:
    """Startup sweep: outputs, page cache, uploads and render temp dirs left by a crash or sudden shutdown."""
    for d in (STATIC_DIR, CACHE_DIR, UPLOAD_DIR):
        n = clear_dir_contents(d)
        if n:
            logger.info(f"[Cleanup] removed {n} leftover item(s) from {d}")
    import glob
    for d in glob.glob(os.path.join(tempfile.gettempdir(), "pdf_render_*")):
        shutil.rmtree(d, ignore_errors=True)


# ── Last result kept for the box editor: only the pages the user changed are re-rendered, the other
# pages are taken from here. Rendered page JPEGs + bubble data of ONE result; dropped when a new
# translation starts, replaced by the next edit, swept on startup (CACHE_DIR). KEEP_LAST_RESULT=0 disables.
KEEP_LAST_RESULT = os.getenv("KEEP_LAST_RESULT", "1") != "0"
LAST_RESULT_DIR = os.path.join(CACHE_DIR, "last_result")
last_result = {}   # {"task_id", "dir", "pages": {page: jpg path}, "data": {page: {"page","w","h","blocks"}}}


def drop_last_result() -> None:
    d = last_result.get("dir")
    last_result.clear()
    if d:
        shutil.rmtree(d, ignore_errors=True)


def store_last_result(task_id: str, pages: dict, data: dict) -> None:
    """Copy the rendered pages into their own folder (the render temp dir is deleted at task end)
    and replace the previous result."""
    old_dir = last_result.get("dir")
    new_dir = os.path.join(LAST_RESULT_DIR, task_id)
    os.makedirs(new_dir, exist_ok=True)
    kept = {}
    for p, src in pages.items():
        if not os.path.exists(src):
            continue
        dst = os.path.join(new_dir, f"page_{int(p):04d}.jpg")
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copyfile(src, dst)
        kept[int(p)] = dst
    last_result.clear()
    last_result.update({"task_id": task_id, "dir": new_dir, "pages": kept, "data": dict(data)})
    if old_dir and os.path.abspath(old_dir) != os.path.abspath(new_dir):
        shutil.rmtree(old_dir, ignore_errors=True)


DOWNLOAD_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".zip": "application/zip",
    ".json": "application/json",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".txt": "text/plain; charset=utf-8",
    ".html": "text/html; charset=utf-8",
}


def strip_runtime_keys(obj):
    """Drop runtime-only keys (leading "_", e.g. _text_mask_png bytes) before JSON export."""
    if isinstance(obj, dict):
        return {k: strip_runtime_keys(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [strip_runtime_keys(v) for v in obj]
    return obj


def apply_language_check(items, engine, source_lang, target_lang, context_chunk_size=12):
    """
    Language check after the LLM (core/text_check.py). items: [(page_num, block)], each block with
    "translated_text" set. Lines with kana / foreign letters are re-translated (max 2x, with a hint),
    then whatever is still wrong is removed; the blocks are fixed in place.
    Returns the report (one dict per touched line, see text_check.check_and_fix).
    """
    from core.text_check import check_and_fix
    entries = [{"page": p, "id": b.get("id"),
                "raw": f"{b.get('text', '')} {b.get('cleaned_text', '')}",
                "text": b.get("translated_text", "") or "", "block": b} for p, b in items]

    def retranslate(bad, hint):
        temp = [{"id": i, "text": e["block"].get("text", "")} for i, e in enumerate(bad)]
        results, _ = engine.translate_batch(temp, source_lang=source_lang, target_lang=target_lang,
                                            context_chunk_size=context_chunk_size, extra_hint=hint)
        return results

    report = check_and_fix(entries, retranslate, source_lang, target_lang)
    for e in entries:
        if e["text"] != (e["block"].get("translated_text", "") or ""):
            e["block"]["translated_text"] = e["text"]
            e["block"]["translation_engine"] = (e["block"].get("translation_engine") or "") + " + 语言检查"
    for r in report:
        logger.info(f"[LangCheck] page {r['entry']['page']} #{r['entry']['id']}: {r['action']} "
                    f"'{r['before']}' -> '{r['after']}'")
    return report


def build_failed_pages_warning(failed_pages) -> str:
    """Human-readable warning for pages whose extraction failed (BUG.md B4). Empty if none."""
    if not failed_pages:
        return ""
    nums = ", ".join(str(p) for p, _ in sorted(failed_pages))
    return f"{len(failed_pages)} 页提取失败，已保留原页未翻译: 第 {nums} 页"


async def preload_all_models():
    """
    Preloads all OCR models and warms up the Turbovec LLM in memory during server startup.
    This guarantees zero cold-start delay when the user uploads their first manga PDF.
    """
    logger.info("⚡ [Preloader] Pre-warming models: MangaOCR, EasyOCR, and Turbovec LLM...")
    
    # 1. Preload & verify Turbovec LLM
    from core.engine import ensure_turbovec_llm_ready
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, ensure_turbovec_llm_ready, True, 25)
    
    # 2. Preload the models the first job actually uses (primary chain first, so a job that
    #    starts right after boot finds them ready). Measured cold loads: OCR ~23s, LaMa ~12s.
    #    EasyOCR is only a fallback and is left lazy.
    def _warm_models():
        import numpy as np
        from PIL import Image
        from core import extractor as ex
        from core.renderer import preload_lama, lama_uses_gpu
        steps = [
            ("placement", ex.prepare_extract_devices),
            ("comic_detector", ex._get_comic_detector),
            ("manga_ocr", ex._get_manga_ocr),
            ("paddle_ocr", ex._get_paddle_ocr),
            ("lama", preload_lama),
        ]
        for name, fn in steps:
            # GPU models are NOT kept resident at boot: next to the LLM they overflow the 4 GB
            # card and the LLM slows down ~4x (B28). They load in 1-2 s when a job needs them.
            if name == "lama" and lama_uses_gpu():
                logger.info("[Preloader] lama: GPU backend, loaded on demand when rendering starts")
                continue
            if name in ("comic_detector", "manga_ocr", "paddle_ocr") and ex._PLACEMENT.get(name) == "gpu":
                logger.info(f"[Preloader] {name}: placed on GPU, loaded on demand when extraction starts")
                continue
            t = time.time()
            try:
                obj = fn()
                if name == "manga_ocr" and obj is not None:
                    obj(Image.fromarray(np.full((64, 64, 3), 255, dtype=np.uint8)))  # warm-up pass
                logger.info(f"✔ [Preloader] {name} ready in {time.time() - t:.1f}s")
            except Exception as ex_err:
                logger.warning(f"⚠️ [Preloader] {name} preload failed: {ex_err}")

    await loop.run_in_executor(None, _warm_models)
    logger.info("✔ [Preloader] All pipeline models preloaded and warm in memory!")


@app.on_event("startup")
async def startup_event():
    global translation_engine
    logger.info("⚡ System Booting... Initializing Pipeline Components...")
    clear_leftovers()
    
    # Try GPU speculative decoding, fallbacks gracefully to standard GPU FP16 or CPU Heuristics
    translation_engine = HighPerformanceTranslationEngine(use_gpu=True)
    
    # Start the non-blocking background queue task listener
    asyncio.create_task(translation_worker())
    logger.info("[✓] Background Translation Queue Guardian started successfully!")

    # Preload and pre-warm all OCR models & Turbovec LLM in the background
    asyncio.create_task(preload_all_models())

# ----------------------------------------------------
# Asynchronous Background Queue Worker Pipeline
# ----------------------------------------------------
# ── Review before render ("先调框"): pause after extraction + translation, the user fixes boxes / text
# in the box editor, then the pages are rendered once with the corrected boxes ──
REVIEW_TIMEOUT_SEC = int(os.getenv("REVIEW_TIMEOUT_SEC", "3600"))   # nobody answers: continue unchanged
review_waits = {}      # task_id -> {"event": asyncio.Event, "payload": {page: rows} | None}


from core.document_skill import DIRECTIONS  # noqa: E402
from core.fonts import list_fonts, normalize_font_id, resolve_font  # noqa: E402


def normalize_direction(value) -> str:
    """Lettering direction of the translation: horizontal (default), vertical (columns left -> right),
    vertical_rtl (columns right -> left) or auto (per box, from how the original text runs)."""
    return value if value in DIRECTIONS else "horizontal"


MOVE_TOL_PT = 1.0      # a box that moved / resized by more than this (PDF points) counts as changed


def build_edit_pages(layout_pages: list, tmap: dict, default_dir: str = "horizontal") -> list:
    """Bubbles + translations of the given layout pages, in the shape the box editor loads."""
    out = []
    for p_data in layout_pages:
        out.append({
            "page": p_data["page_num"],
            "w": float(p_data.get("page_width") or 0), "h": float(p_data.get("page_height") or 0),
            "blocks": [{"id": blk.get("id"), "bbox": [float(v) for v in blk.get("bbox")],
                        "raw": blk.get("cleaned_text", blk.get("text", "")),
                        "text": tmap.get((p_data["page_num"], blk.get("id")), ""),
                        "direction": blk.get("direction") or default_dir, "font": blk.get("font") or "",
                        # lettering size: chosen by the user (0 = automatic), the size it was drawn at,
                        # and the inputs of that layout (original glyph size, bubble) for the live preview
                        "font_size": blk.get("font_size_pt") or 0, "fs_pt": blk.get("fs_pt"),
                        "glyph_pt": blk.get("glyph_pt") or 0, "bubble_pt": blk.get("bubble_pt")}
                       for blk in p_data.get("blocks", []) if blk.get("bbox")],
        })
    return out


def plan_review(pages: list, rows_by_page: dict, tmap: dict, default_dir: str = "horizontal") -> dict:
    """
    Apply the user's box edits to the layout pages IN PLACE (before rendering).
    rows_by_page: {page: [{"id","bbox","text","raw","edited"}]} (only pages the user changed).
    A box missing from the rows was deleted. A box that moved / was resized while its translation was
    left as it was, and a new box with no text at all, need OCR + translation again -> `need_ocr`.
    Returns {"need_ocr": [(page, block)], "pages": [changed page numbers], "removed": n}.
    """
    from core.document_skill import blocks_from_corrections
    by_num = {p["page_num"]: p for p in pages}
    need, changed, removed = [], [], 0
    for num in sorted(rows_by_page):
        page = by_num.get(num)
        if page is None:
            continue
        old = {b["id"]: b for b in page.get("blocks", [])}
        new_blocks = []
        for r in rows_by_page[num]:
            blk = blocks_from_corrections([r])[0]
            o = old.get(r["id"])
            if o is not None:
                blk["text"] = o.get("text", blk["text"])
                blk["cleaned_text"] = o.get("cleaned_text", blk["text"])
                blk["ocr_engine"] = o.get("ocr_engine", blk["ocr_engine"])
                moved = any(abs(a - b) > MOVE_TOL_PT for a, b in zip(r["bbox"], o["bbox"]))
                typed = r["text"] != tmap.get((num, r["id"]), "")
                if moved and not typed:
                    need.append((num, blk))
                turned = (r.get("direction") or default_dir) != (o.get("direction") or default_dir)
                sized = float(r.get("font_size") or 0) != float(o.get("font_size_pt") or 0)
                refont = (r.get("font") or "") != (o.get("font") or "")
                if moved or typed or turned or sized or refont:
                    blk["user_edited"] = True
            else:   # drawn by the user
                blk["user_edited"] = True
                if not r["text"].strip() and not r["raw"].strip():
                    need.append((num, blk))
            tmap[(num, r["id"])] = r["text"]
            new_blocks.append(blk)
        gone = set(old) - {r["id"] for r in rows_by_page[num]}
        for bid in gone:
            tmap.pop((num, bid), None)
        removed += len(gone)
        page["blocks"] = new_blocks
        changed.append(num)
    return {"need_ocr": need, "pages": changed, "removed": removed}


def review_reocr(items: list, pdf_path: str, source_lang: str, target_lang: str, engine, tmap: dict,
                 context_chunk_size: int = 12) -> list:
    """
    OCR + translate (+ language check) the boxes in `items` [(page, block)], in place, and store the
    translations in `tmap`. Returns [(page, block id)] of boxes where no text was recognised (they
    keep the original lettering).
    """
    from core.extractor import ocr_region
    blank, todo = [], []
    for num, blk in items:
        raw = ""
        try:
            raw = (ocr_region(pdf_path, num, blk["bbox"], source_lang) or "").strip()
        except Exception as ex:
            logger.warning(f"[Review] OCR failed on page {num} #{blk['id']}: {ex}")
        if not raw:
            blk["text"] = blk["cleaned_text"] = ""
            tmap[(num, blk["id"])] = ""
            blank.append((num, blk["id"]))
            continue
        blk["text"] = blk["cleaned_text"] = raw
        todo.append((num, blk))
    if todo:
        combined = [{"id": i, "text": blk["text"]} for i, (_, blk) in enumerate(todo)]
        engine.translate_batch(combined, source_lang=source_lang, target_lang=target_lang,
                               context_chunk_size=context_chunk_size)
        for i, (num, blk) in enumerate(todo):
            blk["translated_text"] = combined[i].get("translated_text", "") or ""
        try:
            apply_language_check(todo, engine, source_lang, target_lang, context_chunk_size)
        except Exception as ex:
            logger.warning(f"[Review] language check skipped: {ex}")
        for num, blk in todo:
            tmap[(num, blk["id"])] = blk.get("translated_text", "") or ""
    return blank


async def translation_worker():
    """
    Main loop monitoring translation_queue. Runs tasks asynchronously.
    """
    while True:
        task = await translation_queue.get()
        task_id = task["task_id"]
        pdf_path = task["pdf_path"]
        source_lang = task["source_lang"]
        target_lang = task["target_lang"]
        filename = task["filename"]
        # Per-manga term dictionary (data/terms/<series>.json): no names leak between series
        try:
            from core.engine import set_active_series
            set_active_series(task.get("series") or filename)
        except Exception as _te:
            logger.warning(f"[Terms] cannot load series terms: {_te}")
        page_range = task.get("page_range", "")
        # Re-insert mode: {page_num: [rows]} from a corrected Excel script; no LLM is used
        corrections = task.get("corrections")
        unmatched_rows = {}   # re-insert: {page_num: corrected rows that matched no bubble}
        language_fixes = []   # language check reports (LLM translations that had to be fixed)
        review_first = bool(task.get("review_first")) and corrections is None   # pause before rendering
        review_notes = []     # review: messages for the completion warning
        flagged_rows = []     # re-insert: corrected rows with foreign letters / symbols (only warned)
        
        status_db[task_id] = {
            "percent": 5,
            "stage": "等待队列调度",
            "status": "processing",
            "metrics": None
        }
        
        temp_dir = None
        try:
            if task_id in cancelled_tasks:
                # Cancelled while still queued: do nothing (and do not touch earlier downloads)
                logger.info(f"Task {task_id} was cancelled before it started; skipping.")
                status_db[task_id] = {
                    "percent": 0,
                    "stage": "用户已终止本次翻译任务。",
                    "status": "failed",
                    "message": "Task cancelled by user."
                }
                continue

            logger.info(f"🚀 Processing Task {task_id} inside background loop...")
            
            if source_lang == "Japanese" and not corrections:
                from core.engine import ensure_turbovec_llm_ready
                await asyncio.to_thread(ensure_turbovec_llm_ready, True, 20)
            
            # Outputs of the previous task are dropped once a new task starts (downloaded ones are already gone)
            clear_dir_contents(STATIC_DIR)
            close_edit_sessions()
            # Box editor re-render: only the changed pages are processed, the rest comes from the last result
            base_result = None
            if task.get("base_task_id"):
                if last_result.get("task_id") != task["base_task_id"]:
                    raise Exception("上一次的翻译结果已过期（服务器已清理），请重新翻译。")
                base_result = {"pages": dict(last_result["pages"]), "data": dict(last_result["data"])}
            else:
                drop_last_result()
            
            # Step 1: Intelligent Layout Extraction (CPU Parallel)
            status_db[task_id] = {
                "percent": 15,
                "stage": "提取文档几何布局与分栏排序中",
                "status": "processing"
            }
            await asyncio.sleep(0.3)
            
            import fitz
            src_doc = fitz.open(pdf_path)
            actual_total_pages = len(src_doc)
            src_doc.close()
            
            selected_pages = sorted(list(parse_page_range(page_range, actual_total_pages)))

            extractor = PDFLayoutExtractor(pdf_path)

            # Put extraction models on the GPU only if the free VRAM (next to the LLM) allows it
            from core.extractor import prepare_extract_devices, reset_extract_stats, get_extract_stats
            extract_devices = await asyncio.to_thread(prepare_extract_devices)
            reset_extract_stats()
            
            # Step 2: Streaming Pipeline Setup
            status_db[task_id] = {
                "percent": 15,
                "stage": "并发就绪:",
                "status": "processing"
            }
            await asyncio.sleep(0.3)
            
            # Calculate MD5 hash of the PDF to use for cache directory
            def get_pdf_hash(path):
                hasher = hashlib.md5()
                with open(path, 'rb') as f:
                    for chunk in iter(lambda: f.read(1024*1024), b''):
                        hasher.update(chunk)
                return hasher.hexdigest()
            
            pdf_hash = await asyncio.to_thread(get_pdf_hash, pdf_path)
            layout_data = []
            total_selected_pages = len(selected_pages)

            translated_text_map = {}
            total_metrics = {
                "tokens_per_sec": 0.0,
                "acceptance_rate": 0.0,
                "latency_ms": 0.0,
                "tokens_generated": 0
            }
            
            # Create a temp dir for rendered page JPEGs
            temp_dir = tempfile.mkdtemp(prefix="pdf_render_")
            
            ink_thresh = int(task.get("ink_thresh", 95))
            dilate_iter = int(task.get("dilate_iter", 2))
            max_stroke_ratio = float(task.get("max_stroke_ratio", 0.35))
            font_scale = float(task.get("font_scale", 1.0))

            text_direction = normalize_direction(task.get("text_direction"))
            font_name = normalize_font_id(task.get("font_name"))
            renderer = PDFLayoutRenderer(
                pdf_path,
                None,
                ink_thresh=ink_thresh,
                dilate_iter=dilate_iter,
                max_stroke_ratio=max_stroke_ratio,
                font_scale=font_scale,
                text_direction=text_direction,
                font_path=resolve_font(font_name)
            )
            
            # Semaphore to restrict GPU inpainting to 1 concurrent task to guarantee 4GB VRAM safety
            from core.renderer import LAMA_PARALLEL, reset_render_stats, get_render_stats, preload_lama, lama_uses_gpu
            render_sem = asyncio.Semaphore(LAMA_PARALLEL)
            reset_render_stats()
            lama_preload = None

            def start_lama_preload():
                """Load LaMa in the background once (cold load ~seconds, stalls rendering)."""
                nonlocal lama_preload
                if lama_preload is None:
                    lama_preload = asyncio.ensure_future(asyncio.to_thread(preload_lama))
                return lama_preload
            render_tasks = {}
            src_doc = fitz.open(pdf_path)
            
            # Progress counters
            extracted_count = 0
            translated_count = 0
            rendered_count = 0
            progress_lock = asyncio.Lock()
            
            def update_progress():
                ex_part = int(10.0 * extracted_count / max(1, total_selected_pages))
                trans_part = int(30.0 * translated_count / max(1, total_selected_pages))
                render_part = int(45.0 * rendered_count / max(1, total_selected_pages))
                percent = 5 + ex_part + trans_part + render_part
                
                model_str = ""
                if total_metrics.get("model_usage"):
                    model_str = " | 模型: " + ", ".join(f"{k}:{v}" for k, v in total_metrics["model_usage"].items())

                status_db[task_id]["percent"] = percent
                status_db[task_id]["stage"] = (
                    f"并发中: {extracted_count}/{total_selected_pages} 页, "
                    f"已翻译 {translated_count}/{total_selected_pages} 页, "
                    f"已重绘 {rendered_count}/{total_selected_pages} 页" + model_str
                )
            
            temp_paths = {}
            batch_count = 0

            # ── 3-Stage Asynchronous Overlapped Pipeline ──
            # Stage 1: Extractor Producer (DirectML GPU + CPU)
            # Stage 2: Translator Worker (GPU Qwen 3.5 4B via asyncio.to_thread)
            # Stage 3: Renderer Worker (Hybrid Telea / LaMa Inpainting via worker threads)
            # Orchestration lives in core.async_stages (deadlock-safe, see BUG.md B1).
            from core.async_stages import run_three_stage_pipeline, PipelineCancelled

            async def extracted_pages():
                nonlocal extracted_count
                async for chunk_page in extractor.extract_layout_stream(page_range_list=selected_pages, source_lang=source_lang,
                                                                       mask_only=corrections is not None):
                    if corrections is not None:
                        # Re-insert: no OCR, the bubbles are the Excel rows (positions + text)
                        from core.document_skill import blocks_from_corrections
                        chunk_page["blocks"] = blocks_from_corrections(corrections.get(chunk_page["page_num"], []))
                    layout_data.append(chunk_page)
                    async with progress_lock:
                        extracted_count += 1
                        update_progress()
                    yield chunk_page
                _mark("extract_done")

            # ── Stage timing (for comparing pipeline modes) ──
            stage_times = {}
            t_pipeline0 = time.time()

            def _mark(name):
                stage_times[name] = round(time.time() - t_pipeline0, 1)

            async def translate_pages(pages, context_chunk_size=12):
                """
                Translate the blocks of several pages in ONE engine call so the LLM sees the
                dialogue across page boundaries. Returns the pages that still need rendering.
                """
                nonlocal batch_count, translated_count, rendered_count
                to_render = []
                combined_blocks = []
                block_refs = {}
                for page in pages:
                    to_render.append(page)
                    for block in page.get("blocks", []):
                        u_id = len(combined_blocks)
                        block_copy = dict(block)
                        block_copy["id"] = u_id
                        block_copy["page_num"] = page["page_num"]
                        combined_blocks.append(block_copy)
                        block_refs[u_id] = (page["page_num"], block, block["id"])

                if corrections is not None:
                    # Re-insert: the corrected Excel rows are the translation (matched by bbox)
                    from core.document_skill import match_corrections
                    for page in to_render:
                        p_num = page["page_num"]
                        missed = []
                        texts = match_corrections(page.get("blocks", []), corrections.get(p_num, []), missed)
                        if missed:
                            unmatched_rows[p_num] = unmatched_rows.get(p_num, 0) + len(missed)
                            logger.warning(f"[Re-insert] page {p_num}: {len(missed)} corrected rows matched no "
                                           f"bubble: {[r['id'] for r in missed]}")
                        for block in page.get("blocks", []):
                            block["translated_text"] = texts.get(block["id"], "")
                            block["translation_engine"] = "人工校对 (Excel)"
                            translated_text_map[(p_num, block["id"])] = block["translated_text"]
                    from core.text_check import flag_entries
                    for page in to_render:
                        flagged_rows.extend(flag_entries([
                            {"page": page["page_num"], "id": b["id"], "raw": b.get("text", ""),
                             "text": b.get("translated_text", "")} for b in page.get("blocks", [])]))
                elif combined_blocks:
                    # Offload synchronous translation to thread pool so event loop is never frozen
                    translations, metrics = await asyncio.to_thread(
                        translation_engine.translate_batch,
                        combined_blocks,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        context_chunk_size=context_chunk_size
                    )

                    total_metrics["tokens_per_sec"] += metrics.get("tokens_per_sec", 0.0)
                    total_metrics["acceptance_rate"] += metrics.get("acceptance_rate", 0.0)
                    total_metrics["latency_ms"] += metrics.get("latency_ms", 0.0)
                    total_metrics["tokens_generated"] += metrics.get("tokens_generated", 0)
                    batch_count += 1

                    from core.engine import merge_stats
                    merge_stats(total_metrics.setdefault("translate_breakdown", {}), metrics.get("time_breakdown"))

                    if "model_breakdown" in metrics:
                        if "model_usage" not in total_metrics:
                            total_metrics["model_usage"] = {}
                        for k, v in metrics["model_breakdown"].items():
                            total_metrics["model_usage"][k] = total_metrics["model_usage"].get(k, 0) + v

                    for temp_block in combined_blocks:
                        page_num, orig_block, orig_block_id = block_refs[temp_block["id"]]
                        orig_block["translated_text"] = temp_block.get("translated_text", "")
                        orig_block["is_sfx"] = temp_block.get("is_sfx", False)
                        orig_block["google_trans"] = temp_block.get("google_trans", "")
                        orig_block["cleaned_text"] = temp_block.get("cleaned_text", "")
                        orig_block["ocr_engine"] = temp_block.get("ocr_engine", orig_block.get("ocr_engine", "PaddleOCR (PP-OCRv4)"))
                        orig_block["translation_engine"] = temp_block.get("translation_engine", "Turbovec Qwen 3.5 4B")
                        translated_text_map[(page_num, orig_block_id)] = orig_block["translated_text"]

                    # Language check: kana / foreign letters / stray symbols in the Chinese text
                    try:
                        fixes = await asyncio.to_thread(
                            apply_language_check,
                            [(block_refs[t["id"]][0], block_refs[t["id"]][1]) for t in combined_blocks],
                            translation_engine, source_lang, target_lang, context_chunk_size)
                    except Exception as lc_err:
                        logger.warning(f"[LangCheck] skipped: {lc_err}")
                        fixes = []
                    for fx in fixes:
                        translated_text_map[(fx["entry"]["page"], fx["entry"]["id"])] = fx["entry"]["text"]
                    language_fixes.extend(fixes)

                async with progress_lock:
                    translated_count += len(to_render)
                    update_progress()
                _mark("translate_done")
                return to_render

            async def translate_page(page):
                out = await translate_pages([page])
                return out[0] if out else None

            async def render_page(page):
                nonlocal rendered_count
                p_num = page["page_num"]
                src_page = src_doc[p_num - 1]
                await start_lama_preload()  # no-op once loaded

                dest_path = await renderer.render_single_page_to_temp(
                    page_data=page,
                    src_page=src_page,
                    translated_text_map=translated_text_map,
                    temp_dir=temp_dir,
                    sem=render_sem
                )

                if dest_path and os.path.exists(dest_path):
                    temp_paths[p_num] = dest_path

                async with progress_lock:
                    rendered_count += 1
                    update_progress()
                _mark("render_done")

            async def unload_ocr_models(restart_llm: bool = False):
                from core.extractor import unload_models as unload_ocr
                released = await asyncio.to_thread(unload_ocr)
                _mark("ocr_unloaded")
                # GPU OCR can evict part of the LLM's VRAM to system RAM (4x slower translation):
                # restart the LLM so it is fully resident again before translating (B29).
                # Only where nothing is translating yet (serial): a restart would kill an
                # in-flight request in overlap/stream mode.
                if restart_llm:
                    from core.engine import restart_llm_if_evicted
                    if await asyncio.to_thread(restart_llm_if_evicted):
                        _mark("llm_restarted")
                if not lama_uses_gpu():
                    start_lama_preload()
                logger.info(f"[Pipeline] OCR models released: {sorted(released or [])} "
                            "(GPU ones always; CPU ones only when RAM is low, see OCR_UNLOAD).")

            def hold_gpu_render() -> bool:
                """Called after OCR unload: True = keep rendering until translation is done."""
                from core.gpu_budget import lama_fits_beside_llm
                d = lama_fits_beside_llm()
                stage_times["lama_with_llm"] = bool(d["ok"])
                total_metrics["lama_with_llm"] = d
                logger.info(f"[Pipeline] GPU LaMa beside LLM: {'yes' if d['ok'] else 'no'} ({d['reason']})")
                return not d["ok"]

            def _check_cancel():
                if task_id in cancelled_tasks:
                    raise PipelineCancelled("Task cancelled by user.")

            async def review_pause(pages):
                """Wait for the user to fix boxes / text in the editor (POST /api/v1/translate/review/<id>),
                then apply the edits to `pages` before they are rendered. Cancel and timeout are honoured."""
                wait = {"event": asyncio.Event(), "payload": None}
                review_waits[task_id] = wait
                status_db[task_id] = {
                    "percent": 45, "stage": "翻译完成，等待你调整检测框（调好后点「继续重绘」）", "status": "review",
                    "edit_data": {"pages": build_edit_pages(pages, translated_text_map, text_direction),
                                  "source_lang": source_lang, "text_direction": text_direction,
                                  "font_scale": font_scale, "font_name": font_name},
                    "metrics": total_metrics}
                logger.info(f"[Review] task {task_id}: waiting for the user (timeout {REVIEW_TIMEOUT_SEC}s)")
                t_wait = time.monotonic()
                try:
                    while not wait["event"].is_set():
                        _check_cancel()
                        if time.monotonic() - t_wait > REVIEW_TIMEOUT_SEC:
                            logger.warning(f"[Review] task {task_id}: no answer, continuing unchanged")
                            break
                        try:
                            await asyncio.wait_for(wait["event"].wait(), timeout=0.5)
                        except asyncio.TimeoutError:
                            pass
                finally:
                    review_waits.pop(task_id, None)
                total_metrics["review_wait_s"] = round(time.monotonic() - t_wait, 1)
                rows = wait["payload"] or {}
                status_db[task_id] = {"percent": 45, "stage": "按调整后的检测框重绘中", "status": "processing",
                                      "metrics": total_metrics}
                if not rows:
                    return
                plan = plan_review(pages, rows, translated_text_map, text_direction)
                logger.info(f"[Review] task {task_id}: edited pages {plan['pages']}, removed {plan['removed']} box(es), "
                            f"{len(plan['need_ocr'])} box(es) to recognise again")
                if plan["need_ocr"]:
                    status_db[task_id]["stage"] = f"重新识别并翻译 {len(plan['need_ocr'])} 个改动的框"
                    _check_cancel()
                    blank = await asyncio.to_thread(review_reocr, plan["need_ocr"], pdf_path, source_lang,
                                                    target_lang, translation_engine, translated_text_map,
                                                    context_chunk_size)
                    if blank:
                        review_notes.append("重新识别时没读到文字的框（保留原文）: "
                                            + ", ".join(f"第{p}页#{i}" for p, i in blank))
                status_db[task_id]["stage"] = "按调整后的检测框重绘中"

            async def run_serial():
                """Mode 'serial': extract ALL -> unload OCR -> translate ALL (big context) -> render ALL."""
                # MangaOCR on the GPU in a child process for the extraction stage only (B30);
                # it is stopped by unload_ocr_models() before the LLM translates.
                from core.extractor import start_manga_gpu_worker
                if corrections is None:   # re-insert runs no OCR
                    await asyncio.to_thread(start_manga_gpu_worker, total_selected_pages)
                pages = []
                async for p in extracted_pages():
                    _check_cancel()
                    pages.append(p)
                pages.sort(key=lambda x: x["page_num"])
                await unload_ocr_models(restart_llm=not corrections)
                if not lama_uses_gpu():
                    start_lama_preload()  # CPU is idle while the LLM translates (CPU LaMa only;
                                          # GPU LaMa waits until the LLM is done: B28)

                # Group whole pages so each engine call carries up to context_chunk_size lines
                to_render, group, n_lines = [], [], 0
                for p in pages:
                    n = len(p.get("blocks", []))
                    if group and n_lines + n > context_chunk_size:
                        _check_cancel()
                        to_render += await translate_pages(group, context_chunk_size)
                        group, n_lines = [], 0
                    group.append(p)
                    n_lines += n
                if group:
                    _check_cancel()
                    to_render += await translate_pages(group, context_chunk_size)

                if review_first:
                    await review_pause(pages)

                # Nothing else runs now, so render several pages at once: while one page waits
                # for LaMa, others do masks / Telea / text drawing / JPEG encoding.
                page_sem = asyncio.Semaphore(RENDER_CONCURRENCY)

                async def _render_one(p):
                    async with page_sem:
                        _check_cancel()
                        await render_page(p)

                await asyncio.gather(*[_render_one(p) for p in to_render])

            requested_mode = task.get("pipeline_mode", DEFAULT_PIPELINE_MODE)
            pipeline_mode = resolve_pipeline_mode(requested_mode, total_selected_pages)
            context_chunk_size = max(1, int(task.get("context_chunk_size", DEFAULT_CONTEXT_CHUNK)))
            # stream / overlap: buffer pages until this many lines, then translate them together
            stream_batch_lines = max(0, int(task.get("translate_batch_lines", DEFAULT_STREAM_BATCH_LINES)))
            logger.info(f"[Pipeline] mode={pipeline_mode} context_chunk_size={context_chunk_size} "
                        f"stream_batch_lines={stream_batch_lines if pipeline_mode != 'serial' else '-'}")

            try:
                if pipeline_mode == "serial":
                    await run_serial()
                else:
                    if pipeline_mode == "overlap":
                        # MangaOCR GPU child during extraction (jobs > 20 pages, see
                        # manga_gpu_worker_wanted): it fits
                        # next to the resident LLM (serial 30 p: extract 128 s -> 61.5 s, LLM not
                        # evicted); stopped by unload_ocr_models() when extraction ends.
                        from core.extractor import start_manga_gpu_worker
                        await asyncio.to_thread(start_manga_gpu_worker, total_selected_pages)
                    if lama_uses_gpu() and pipeline_mode == "overlap":
                        # nothing is translating yet: the only safe moment to restart an
                        # evicted LLM in overlap mode (B29)
                        from core.engine import restart_llm_if_evicted
                        if await asyncio.to_thread(restart_llm_if_evicted):
                            _mark("llm_restarted")
                    # 'stream' : extract / translate / render all concurrently (original behaviour)
                    # 'overlap': extract + translate concurrently; when extraction ends, unload OCR,
                    #            then render concurrently with the remaining translation
                    await run_three_stage_pipeline(
                        extracted_pages(),
                        translate_page,
                        render_page,
                        is_cancelled=lambda: task_id in cancelled_tasks,
                        queue_size=3,
                        hold_render_until_source_done=(pipeline_mode == "overlap"),
                        on_source_done=unload_ocr_models if pipeline_mode == "overlap" else None,
                        # Wait until ~context_chunk_size lines are extracted, then translate them
                        # in one LLM call (more dialogue context, fewer calls). 0 = per page.
                        translate_batch=(lambda pages: translate_pages(pages, context_chunk_size))
                        if stream_batch_lines > 0 else None,
                        batch_lines=stream_batch_lines,
                        page_lines=lambda page: len(page.get("blocks", [])),
                        # Several pages at once: CPU stages overlap while one page waits for LaMa
                        render_concurrency=RENDER_CONCURRENCY,
                        # GPU LaMa: next to the translating LLM only if it fits once OCR is
                        # unloaded (fixed-ctx LLM), else wait for translation (B28)
                        hold_render_until_translate_done=hold_gpu_render
                        if lama_uses_gpu() and pipeline_mode == "overlap" else False,
                    )
            except PipelineCancelled:
                raise Exception("Task cancelled by user.")
            except Exception as ex:
                logger.error(f"[Pipeline] Stage error: {ex}", exc_info=True)
                raise Exception(f"流水线处理异常: {ex}")

            stage_times["total"] = round(time.time() - t_pipeline0, 1)
            total_metrics["pipeline_mode"] = pipeline_mode if requested_mode == pipeline_mode else f"{requested_mode}->{pipeline_mode}"
            total_metrics["extract_devices"] = {k: v for k, v in extract_devices.items() if not k.startswith("_")}
            total_metrics["stage_times"] = stage_times
            total_metrics["render_breakdown"] = get_render_stats()
            if lama_uses_gpu():
                # free the GPU for the next job's OCR / translation (LaMa reloads in ~1-2 s)
                from core.renderer import unload_models as unload_lama
                await asyncio.to_thread(unload_lama)
            total_metrics["extract_breakdown"] = get_extract_stats()
            logger.info(f"[Timing] mode={pipeline_mode} pages={total_selected_pages} {stage_times}")


            failed_pages = sorted(extractor.failed_pages)
            if not layout_data:
                detail = f"（失败页: {', '.join(str(p) for p, _ in failed_pages)}；首个错误: {failed_pages[0][1]}）" if failed_pages else ""
                raise Exception("未能成功提取到任何页面数据，请检查 PDF 文件完整性或页面范围设置。" + detail)

            extraction_warning = build_failed_pages_warning(failed_pages)
            if unmatched_rows:
                from core.document_skill import build_unmatched_warning
                extraction_warning = "；".join(w for w in (extraction_warning, build_unmatched_warning(unmatched_rows)) if w)
            from core.text_check import build_language_warning, build_flagged_warning
            extraction_warning = "；".join(w for w in (extraction_warning, build_language_warning(language_fixes),
                                                     build_flagged_warning(flagged_rows), *review_notes) if w)
            if extraction_warning:
                logger.warning(f"Task {task_id}: {extraction_warning}")

            # Retain original sequential reading order for DOC and JSON generation
            layout_data.sort(key=lambda x: x["page_num"])
                
            if batch_count > 0:
                total_metrics["tokens_per_sec"] /= batch_count
                total_metrics["acceptance_rate"] /= batch_count
                total_metrics["latency_ms"] /= batch_count

            # Step 3: CJK Double Check & Sentinel Checks
            if task_id in cancelled_tasks:
                raise Exception("Task cancelled by user.")
                
            status_db[task_id] = {
                "percent": 90,
                "stage": "简体中文校验哨兵防逃逸双重过滤中",
                "status": "processing",
                "metrics": total_metrics
            }
            await asyncio.sleep(0.3)
            
            # Step 4: Final output PDF assembly
            if task_id in cancelled_tasks:
                raise Exception("Task cancelled by user.")
                
            status_db[task_id] = {
                "percent": 95,
                "stage": "编译拼装 PDF 中",
                "status": "processing",
                "metrics": total_metrics
            }
            
            out_filename = f"translated_{uuid.uuid4().hex[:8]}_{filename}"
            output_pdf_path = os.path.join(STATIC_DIR, out_filename)
            
            if base_result:   # pages that were not re-rendered keep their earlier result
                for bp, bpath in base_result["pages"].items():
                    if bp not in temp_paths and os.path.exists(bpath):
                        temp_paths[bp] = bpath
            out_doc = fitz.open()
            for page_num in range(1, actual_total_pages + 1):
                src_page = src_doc[page_num - 1]
                if page_num in temp_paths and os.path.exists(temp_paths[page_num]):
                    out_page = out_doc.new_page(
                        width=src_page.rect.width,
                        height=src_page.rect.height
                    )
                    out_page.insert_image(out_page.rect, filename=temp_paths[page_num])
                else:
                    out_doc.insert_pdf(src_doc, from_page=page_num - 1, to_page=page_num - 1)
            
            src_doc.close()
            out_doc.save(output_pdf_path, garbage=4, deflate=True)
            out_doc.close()
            
            # --- Bilingual proofreading script (Excel); can be edited and re-inserted ---
            script_pages = []
            for p_data in layout_data:
                p_num = p_data["page_num"]
                if p_num not in selected_pages: continue
                p_blocks = [{
                    "id": blk.get("id"),
                    "raw": blk.get("cleaned_text", blk.get("text", "")),
                    # Key is (page_num, block_id) tuple — must match how translated_text_map is built
                    "translated": translated_text_map.get((p_num, blk.get("id")), ""),
                    "bbox": blk.get("bbox"),
                } for blk in p_data.get("blocks", [])]
                if p_blocks:
                    script_pages.append({"page_num": p_num, "blocks": p_blocks})

            # Bubbles + translations for the box editor (page image comes from /api/v1/edit/...)
            edit_pages = build_edit_pages([p for p in layout_data if p["page_num"] in selected_pages],
                                          translated_text_map, text_direction)
            if base_result:   # earlier pages stay in the Excel script and stay editable
                done = {ep["page"] for ep in edit_pages}
                for bp, bdata in sorted(base_result["data"].items()):
                    if bp in done:
                        continue
                    edit_pages.append(bdata)
                    script_pages.append({"page_num": bp, "blocks": [
                        {"id": b["id"], "raw": b["raw"], "translated": b["text"], "bbox": b["bbox"]}
                        for b in bdata["blocks"]]})
                edit_pages.sort(key=lambda ep: ep["page"])
                script_pages.sort(key=lambda sp: sp["page_num"])
            edit_data = {"pages": edit_pages, "source_lang": source_lang, "text_direction": text_direction,
                         "font_scale": font_scale, "font_name": font_name}

            xlsx_path = None
            try:
                from core.document_skill import LocalDocumentSkill
                xlsx_path = LocalDocumentSkill(STATIC_DIR).generate_bilingual_xlsx(
                    task_id, script_pages, filename,
                    meta={"pdf_md5": pdf_hash, "source_lang": source_lang, "ink_thresh": ink_thresh,
                          "dilate_iter": dilate_iter, "max_stroke_ratio": max_stroke_ratio,
                          "font_scale": font_scale})
            except Exception as e:
                logger.error(f"Failed to generate Excel script: {e}")

            # Generate JSON Output for Layout & Extracted Data
            out_json_filename = f"translated_{uuid.uuid4().hex[:8]}_{filename}.json"
            output_json_path = os.path.join(STATIC_DIR, out_json_filename)
            try:
                import json
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(strip_runtime_keys(layout_data), f, ensure_ascii=False, indent=4)
            except Exception as e:
                logger.error(f"Failed to generate JSON output: {e}")

            # Pack ZIP (CBZ) Archive
            base_filename_no_ext = os.path.splitext(filename)[0]
            out_zip_filename = f"translated_{uuid.uuid4().hex[:8]}_{base_filename_no_ext}.zip"
            output_zip_path = os.path.join(STATIC_DIR, out_zip_filename)
            
            def write_zip():
                with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_f:
                    for page_num in range(1, actual_total_pages + 1):
                        if page_num in temp_paths and os.path.exists(temp_paths[page_num]):
                            arcname = f"page_{page_num:03d}.jpg"
                            zip_f.write(temp_paths[page_num], arcname)
                    if xlsx_path and os.path.exists(xlsx_path):
                        zip_f.write(xlsx_path, os.path.basename(xlsx_path))
                    if os.path.exists(output_json_path):
                        zip_f.write(output_json_path, os.path.basename(output_json_path))
                            
            await asyncio.to_thread(write_zip)
            
            download_url = f"/api/v1/download/{out_filename}"
            download_zip_url = f"/api/v1/download/{out_zip_filename}"
            download_xlsx_url = f"/api/v1/download/{os.path.basename(xlsx_path)}" if xlsx_path else ""
            download_json_url = f"/api/v1/download/{out_json_filename}"
            
            if KEEP_LAST_RESULT:
                try:
                    await asyncio.to_thread(store_last_result, task_id, dict(temp_paths),
                                            {ep["page"]: ep for ep in edit_pages})
                except Exception as keep_err:
                    logger.warning(f"[Editor] could not keep the result for later edits: {keep_err}")
                    drop_last_result()
            logger.info(f"✅ Task {task_id} completed successfully.")
            
            status_db[task_id] = {
                "percent": 100,
                "stage": "处理完成！" + (f"（{extraction_warning}）" if extraction_warning else ""),
                "status": "complete",
                "warning": extraction_warning,
                "failed_pages": [p for p, _ in failed_pages],
                "download_url": download_url,
                "download_zip_url": download_zip_url,
                "download_xlsx_url": download_xlsx_url,
                "download_json_url": download_json_url,
                "edit_data": edit_data,
                "metrics": total_metrics
            }
            logger.info(f"[✓] Task {task_id} completed successfully! Download URL: {download_url}")
            
        except Exception as e:
            logger.error(f"[!] Critical failure while processing task {task_id}: {e}")
            status_db[task_id] = {
                "percent": 0,
                "stage": f"发生管道崩溃错误: {str(e)}",
                "status": "failed",
                "message": str(e)
            }
        finally:
            # Rendered JPEGs never outlive the task (success, failure or cancel)
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
            if os.path.exists(pdf_path):
                try:
                    os.remove(pdf_path)
                except Exception as ex:
                    logger.warning(f"Could not clean file {pdf_path}: {ex}")
            cancelled_tasks.discard(task_id)
            translation_queue.task_done()

# ----------------------------------------------------
# REST API Controllers & Web Handlers
# ----------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_index_dashboard():
    """
    Renders and serves the beautiful premium Cyber Dark Dashboard UI.
    """
    html_path = os.path.join(BASE_DIR, "templates/index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    else:
        raise HTTPException(status_code=404, detail="Index templates not found.")

# Session memory database for interactive tuning preview
preview_sessions = {}
preview_lock = asyncio.Lock()

@app.post("/api/v1/preview/render")
async def preview_render_page(
    file: UploadFile = File(None),
    preview_id: str = Form(None),
    page_num: int = Form(1),
    ink_thresh: int = Form(95),
    dilate_iter: int = Form(2),
    max_stroke_ratio: float = Form(0.35),
    font_scale: float = Form(1.0),
    source_lang: str = Form("Japanese")
):
    """
    Lightning-fast interactive preview rendering endpoint for tuning sidebar:
    1. First call with file: uploads PDF, creates preview_id, extracts requested page layout & text.
    2. Subsequent calls with preview_id: uses cached layout and text, only re-runs inpainting & typography in ~25ms!
    Returns original, mask overlay (highlighting ink in magenta), and rendered result base64 images.
    """
    global translation_engine
    import time
    t0 = time.time()
    
    try:
        async with preview_lock:
            if file is not None and file.filename:
                if not file.filename.endswith(".pdf"):
                    return {"success": False, "error": "Only PDF files are supported for preview."}
                preview_id = uuid.uuid4().hex[:10]
                temp_preview_path = os.path.join(UPLOAD_DIR, f"preview_{preview_id}_{file.filename}")
                try:
                    content = await file.read()
                    with open(temp_preview_path, "wb") as f:
                        f.write(content)
                except Exception as ex:
                    return {"success": False, "error": f"Failed to save preview upload: {ex}"}
                
                try:
                    doc = fitz.open(temp_preview_path)
                    total_pages = len(doc)
                    doc.close()
                except Exception as ex:
                    return {"success": False, "error": f"Invalid PDF file: {ex}"}

                # Bound cache size to prevent memory/disk bloat
                if len(preview_sessions) >= 8:
                    oldest_id = min(preview_sessions.keys(), key=lambda k: preview_sessions[k].get("last_accessed", 0))
                    old_info = preview_sessions.pop(oldest_id, None)
                    if old_info and os.path.exists(old_info.get("pdf_path", "")):
                        try:
                            os.remove(old_info["pdf_path"])
                        except Exception:
                            pass

                preview_sessions[preview_id] = {
                    "pdf_path": temp_preview_path,
                    "total_pages": total_pages,
                    "pages": {},
                    "translated_text_map": {},
                    "last_accessed": time.time()
                }
            elif preview_id:
                if preview_id not in preview_sessions:
                    return {"success": False, "error": "Preview session expired or not found. Please re-select the file."}
                preview_sessions[preview_id]["last_accessed"] = time.time()
            else:
                return {"success": False, "error": "Either file or preview_id must be provided."}

        sess = preview_sessions[preview_id]
        pdf_path = sess["pdf_path"]
        total_pages = sess["total_pages"]
        page_num = max(1, min(int(page_num), total_pages))

        # 1. Ensure page layout is extracted (cached per session)
        if page_num not in sess["pages"]:
            extractor = PDFLayoutExtractor(pdf_path)
            page_data = await asyncio.to_thread(extractor._extract_single_page, pdf_path, page_num, source_lang)
            sess["pages"][page_num] = page_data
        else:
            page_data = sess["pages"][page_num]

        # 2. Ensure text blocks are translated (cached per session)
        if translation_engine is None:
            translation_engine = HighPerformanceTranslationEngine()

        untranslated_blocks = []
        for b in page_data.get("blocks", []):
            b_key = (page_num, b["id"])
            if b_key not in sess["translated_text_map"]:
                raw_text = b.get("text", "").strip()
                if raw_text:
                    untranslated_blocks.append(b)
                else:
                    sess["translated_text_map"][b_key] = ""

        if untranslated_blocks:
            try:
                await asyncio.to_thread(translation_engine.translate_batch, untranslated_blocks, source_lang=source_lang)
                for b in untranslated_blocks:
                    sess["translated_text_map"][(page_num, b["id"])] = b.get("translated_text", "")
            except Exception as trans_ex:
                logger.warning(f"Preview translation fallback: {trans_ex}")
                for b in untranslated_blocks:
                    sess["translated_text_map"][(page_num, b["id"])] = b.get("text", "")

        # 3. Render preview images using tuned parameters
        doc = fitz.open(pdf_path)
        src_page = doc[page_num - 1]

        renderer = PDFLayoutRenderer(
            original_pdf_path=pdf_path,
            ink_thresh=ink_thresh,
            dilate_iter=dilate_iter,
            max_stroke_ratio=max_stroke_ratio,
            font_scale=font_scale
        )

        t_render = time.time()
        preview_output = await asyncio.to_thread(
            renderer.render_preview_images,
            page_data,
            src_page,
            sess["translated_text_map"]
        )
        doc.close()
        render_time_ms = int((time.time() - t_render) * 1000)
        total_time_ms = int((time.time() - t0) * 1000)

        detected_blocks = []
        for b in page_data.get("blocks", []):
            b_id = b["id"]
            detected_blocks.append({
                "id": b_id,
                "text": b.get("text", ""),
                "translated": sess["translated_text_map"].get((page_num, b_id), ""),
                "bbox": b.get("bbox", [])
            })

        return {
            "success": True,
            "preview_id": preview_id,
            "page_num": page_num,
            "total_pages": total_pages,
            "original_img": preview_output["original"],
            "mask_img": preview_output["mask"],
            "result_img": preview_output["result"],
            "blocks_count": preview_output["blocks_count"],
            "blocks": detected_blocks,
            "render_time_ms": render_time_ms,
            "total_time_ms": total_time_ms,
            "params": {
                "ink_thresh": ink_thresh,
                "dilate_iter": dilate_iter,
                "max_stroke_ratio": max_stroke_ratio,
                "font_scale": font_scale
            }
        }
    except Exception as e:
        logger.error(f"Failed to generate preview: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

# ── Box editor: the browser keeps the source PDF, the server keeps a copy only while editing ──
edit_sessions = {}          # edit_id -> {"pdf_path", "total_pages"}
edit_lock = asyncio.Lock()  # one OCR/translation at a time
MAX_EDIT_PAGE_PX = 1400


def close_edit_sessions() -> None:
    for sid in list(edit_sessions):
        info = edit_sessions.pop(sid, None)
        if info:
            _remove_quietly(info["pdf_path"])


def render_edit_page_png(pdf_path: str, page_num: int, max_px: int = MAX_EDIT_PAGE_PX):
    """(png bytes, page width pt, page height pt) of the ORIGINAL page, longest side <= max_px."""
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_num - 1]
        w, h = float(page.rect.width), float(page.rect.height)
        scale = min(max_px / max(w, h), 3.0)
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        return pix.tobytes("png"), w, h
    finally:
        doc.close()


@app.post("/api/v1/edit/open")
async def edit_open(file: UploadFile = File(...), task_id: str = Form("")):
    """Upload the source PDF once for the box editor; returns edit_id.
    base_available: the server still has the rendered pages of result `task_id`, so an edit can
    re-render only the changed pages."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    close_edit_sessions()
    edit_id = uuid.uuid4().hex[:10]
    path = os.path.join(UPLOAD_DIR, f"edit_{edit_id}.pdf")
    with open(path, "wb") as f:
        f.write(await file.read())
    try:
        doc = fitz.open(path)
        total = len(doc)
        doc.close()
    except Exception as ex:
        _remove_quietly(path)
        raise HTTPException(status_code=400, detail=f"Invalid PDF file: {ex}")
    edit_sessions[edit_id] = {"pdf_path": path, "total_pages": total}
    return {"edit_id": edit_id, "total_pages": total,
            "base_available": bool(task_id) and last_result.get("task_id") == task_id}


def _edit_session(edit_id: str) -> dict:
    info = edit_sessions.get(edit_id)
    if not info:
        raise HTTPException(status_code=404, detail="编辑会话已过期，请重新打开框编辑器。")
    return info


@app.get("/api/v1/edit/{edit_id}/page/{page_num}")
async def edit_page_image(edit_id: str, page_num: int):
    info = _edit_session(edit_id)
    if page_num < 1 or page_num > info["total_pages"]:
        raise HTTPException(status_code=404, detail="Page out of range.")
    png, w, h = await asyncio.to_thread(render_edit_page_png, info["pdf_path"], page_num)
    return Response(content=png, media_type="image/png",
                    headers={"X-Page-Width": f"{w:.2f}", "X-Page-Height": f"{h:.2f}", "Cache-Control": "no-store"})


@app.delete("/api/v1/edit/{edit_id}")
async def edit_close(edit_id: str):
    info = edit_sessions.pop(edit_id, None)
    if info:
        _remove_quietly(info["pdf_path"])
    return {"status": "closed"}


class EditOcrRequest(BaseModel):
    page: int
    bbox: list[float]
    source_lang: str = "Japanese"
    target_lang: str = "Simplified Chinese"


class FontBox(BaseModel):
    id: int
    bbox: list
    text: str = ""
    direction: str = ""
    font: str = ""
    font_size: float = 0
    glyph_pt: float = 0
    bubble_pt: Optional[list] = None
    moved: bool = False


class FontSizeRequest(BaseModel):
    page_w: float
    page_h: float
    text_direction: str = "horizontal"
    font_scale: float = 1.0
    font_name: str = ""
    boxes: List[FontBox]


def estimate_font_sizes(req: "FontSizeRequest") -> dict:
    """
    Size (PDF points) each box will be lettered at, computed with the renderer's own layout code:
    {id: {"pt": size, "estimate": bool}}. A box that was not moved keeps the original glyph size and
    the speech bubble found at the last render, so its number matches that render; a moved / drawn
    box (or a page that was never rendered) only gets an estimate from its rectangle.
    """
    from PIL import Image
    from core.document_skill import clean_font_size, MAX_EDITOR_ROWS, MAX_EDITOR_TEXT
    if not (10 <= req.page_w <= 20000 and 10 <= req.page_h <= 20000) or len(req.boxes) > MAX_EDITOR_ROWS:
        raise ValueError("bad page size or too many boxes")
    scale = min(2.0, 1600.0 / max(1.0, req.page_h))
    W, H = max(1, int(req.page_w * scale)), max(1, int(req.page_h * scale))
    renderer = PDFLayoutRenderer(None, None, font_scale=min(max(float(req.font_scale or 1.0), 0.3), 3.0),
                                 text_direction=normalize_direction(req.text_direction),
                                 font_path=resolve_font(req.font_name))
    items, blocks = [], {}
    for b in req.boxes:
        text = (b.text or "").strip()[:MAX_EDITOR_TEXT]
        if not text or len(b.bbox) != 4:
            continue
        x0, y0, x1, y1 = (max(0, int(float(v) * scale)) for v in b.bbox)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        known = (not b.moved) and (b.glyph_pt > 0 or bool(b.bubble_pt))
        bubble = [int(float(v) * scale) for v in b.bubble_pt] if (not b.moved and b.bubble_pt and len(b.bubble_pt) == 4) else None
        blk = {"id": b.id, "direction": b.direction if b.direction in DIRECTIONS else "", "font": b.font,
               "font_size_pt": clean_font_size(b.font_size), "user_edited": True}
        style = {"glyph_px": int(b.glyph_pt * scale) if not b.moved else 0, "clean_bg": False, "scale": scale,
                 "bubble": bubble, "orient": None}
        items.append((x0, y0, x1, y1, text, blk, style))
        blocks[b.id] = (blk, not known)
    renderer._layout_translations(Image.new("RGB", (W, H), "white"), items)
    return {bid: {"pt": blk.get("fs_pt"), "estimate": est} for bid, (blk, est) in blocks.items()}


@app.get("/api/v1/fonts")
async def get_fonts():
    """Fonts the translation can be lettered with: [{"id", "name"}], "" = automatic."""
    return {"fonts": await asyncio.to_thread(list_fonts)}


@app.post("/api/v1/edit/fontsize")
async def edit_font_sizes(req: FontSizeRequest):
    """Live 'final lettering size' numbers for the box editor (no rendering, no OCR, no LLM)."""
    try:
        sizes = await asyncio.to_thread(estimate_font_sizes, req)
    except ValueError as ex:
        raise HTTPException(status_code=400, detail=str(ex))
    return {"sizes": sizes}


@app.post("/api/v1/edit/{edit_id}/ocr")
async def edit_ocr_translate(edit_id: str, req: EditOcrRequest):
    """OCR + translate ONE box the user drew (no other page is touched)."""
    global translation_engine
    info = _edit_session(edit_id)
    b = req.bbox
    if req.page < 1 or req.page > info["total_pages"]:
        raise HTTPException(status_code=404, detail="Page out of range.")
    if len(b) != 4 or not all(v == v and abs(v) < 1e6 for v in b) or b[2] - b[0] < 4 or b[3] - b[1] < 4:
        raise HTTPException(status_code=400, detail="框太小或坐标无效。")
    try:
        async with edit_lock:
            from core.extractor import ocr_region
            raw = await asyncio.to_thread(ocr_region, info["pdf_path"], req.page, b, req.source_lang)
            raw = (raw or "").strip()
            if not raw:
                return {"raw": "", "translated": "", "note": "这个框里没有识别到文字，请手动输入译文。"}
            if translation_engine is None:
                translation_engine = HighPerformanceTranslationEngine()
            if req.source_lang == "Japanese":
                from core.engine import ensure_turbovec_llm_ready
                await asyncio.to_thread(ensure_turbovec_llm_ready, True, 20)
            block = {"id": 0, "text": raw}
            await asyncio.to_thread(translation_engine.translate_batch, [block],
                                    source_lang=req.source_lang, target_lang=req.target_lang, context_chunk_size=12)
            await asyncio.to_thread(apply_language_check, [(req.page, block)], translation_engine,
                                    req.source_lang, req.target_lang)
            return {"raw": raw, "translated": block.get("translated_text", "") or ""}
    except HTTPException:
        raise
    except Exception as ex:
        logger.error(f"[Editor] OCR/translate failed: {ex}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"识别或翻译失败: {ex}")


async def _load_corrections(xlsx: UploadFile, pdf: UploadFile):
    """Validate and parse a corrected Excel script; the xlsx never stays on disk."""
    from core.document_skill import read_corrections, CorrectionsError, MAX_XLSX_BYTES
    if not xlsx.filename.lower().endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="校对文件必须是 .xlsx（本系统导出的 Excel 台本）")
    data = await xlsx.read()
    if len(data) > MAX_XLSX_BYTES:
        raise HTTPException(status_code=400, detail="Excel 文件过大")
    # Parse from memory: a temp file stayed locked on Windows when openpyxl failed half-way
    # (WinError 32 on unlink hid the real error), and nothing touches the disk this way.
    try:
        meta, rows = read_corrections(io.BytesIO(data))
    except CorrectionsError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not rows:
        raise HTTPException(status_code=400, detail="Excel 台本里没有任何气泡")
    pdf_bytes = await pdf.read()
    await pdf.seek(0)
    if meta.get("pdf_md5") and hashlib.md5(pdf_bytes).hexdigest() != meta["pdf_md5"]:
        want = meta.get("source_file") or "生成这份台本时使用的 PDF"
        hint = "（请上传原始 PDF，不是 translated_ 开头的翻译结果）" if pdf.filename.startswith("translated_") else ""
        raise HTTPException(status_code=400, detail=f"PDF 与 Excel 台本不匹配：应上传「{want}」{hint}")
    return rows, meta


@app.post("/api/v1/translate/upload")
async def upload_pdf_file(
    file: UploadFile = File(...),
    source_lang: str = Form(...),
    target_lang: str = Form("Simplified Chinese"),
    page_range: str = Form(""),
    force_retranslate: bool = Form(True),
    ink_thresh: int = Form(95),
    dilate_iter: int = Form(2),
    max_stroke_ratio: float = Form(0.35),
    font_scale: float = Form(1.0),
    pipeline_mode: str = Form(DEFAULT_PIPELINE_MODE),
    context_chunk_size: int = Form(DEFAULT_CONTEXT_CHUNK),
    translate_batch_lines: int = Form(DEFAULT_STREAM_BATCH_LINES),
    series: str = Form(""),
    corrections: UploadFile = File(None),
    corrections_json: str = Form(""),
    base_task_id: str = Form(""),
    review_first: bool = Form(False),
    text_direction: str = Form("horizontal"),
    font_name: str = Form("")
):
    """
    Uploads the raw PDF file, assigns uuid, applies custom tuning parameters, and queues the task.
    With `corrections` (the Excel script of an earlier run, edited by the user) the task
    re-inserts those translations into the same PDF instead of translating again.
    """
    if not file.filename.endswith(".pdf"):
        return {"error": "Invalid format. Only PDF files are supported."}

    correction_rows = None
    base = ""
    if review_first:
        pipeline_mode = "serial"   # only serial has one point where everything is translated but nothing rendered
    if corrections is not None and corrections.filename:
        correction_rows, meta = await _load_corrections(corrections, file)
        # Same pages and render parameters as the run that produced the script
        page_range = ",".join(str(p) for p in sorted(correction_rows))
        source_lang = meta.get("source_lang") or source_lang
        try:
            ink_thresh = int(float(meta.get("ink_thresh", ink_thresh)))
            dilate_iter = int(float(meta.get("dilate_iter", dilate_iter)))
            max_stroke_ratio = float(meta.get("max_stroke_ratio", max_stroke_ratio))
            font_scale = float(meta.get("font_scale", font_scale))
        except ValueError:
            pass
        pipeline_mode = "serial"
    elif (corrections_json or "").strip():
        # Box editor: edited bubbles (positions + translations) as JSON; render parameters come from the form
        from core.document_skill import parse_corrections_json, CorrectionsError
        try:
            correction_rows = parse_corrections_json(corrections_json)
        except CorrectionsError as e:
            raise HTTPException(status_code=400, detail=str(e))
        page_range = ",".join(str(p) for p in sorted(correction_rows))
        pipeline_mode = "serial"
        if (base_task_id or "").strip():
            # only the pages in corrections_json are re-rendered; the others come from that result
            if last_result.get("task_id") != base_task_id.strip():
                raise HTTPException(status_code=409, detail="上一次的翻译结果已过期（服务器已清理），请重新翻译，或在编辑器里重绘全部页。")
            base = base_task_id.strip()

    task_id = str(uuid.uuid4())
    temp_filename = f"source_{task_id}_{file.filename}"
    temp_filepath = os.path.join(UPLOAD_DIR, temp_filename)
    
    try:
        with open(temp_filepath, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write file on disk: {e}")
        
    task_metadata = {
        "task_id": task_id,
        "pdf_path": temp_filepath,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "filename": file.filename,
        "page_range": page_range,
        "force_retranslate": force_retranslate,
        "ink_thresh": ink_thresh,
        "dilate_iter": dilate_iter,
        "max_stroke_ratio": max_stroke_ratio,
        "font_scale": font_scale,
        "pipeline_mode": pipeline_mode if pipeline_mode in PIPELINE_MODES else DEFAULT_PIPELINE_MODE,
        "context_chunk_size": context_chunk_size,
        "translate_batch_lines": translate_batch_lines,
        # optional: term dictionary name; empty = derived from the file name (volume removed)
        "series": (series or "").strip(),
        "corrections": correction_rows,
        "base_task_id": base,
        # pause after extraction + translation so the user can fix boxes before anything is rendered
        "review_first": bool(review_first) and correction_rows is None,
        # lettering of the translation: "horizontal" (left -> right) or "vertical" (top -> bottom, columns left -> right)
        "text_direction": normalize_direction(text_direction),
        # font of the translation ("" = automatic); only ids from core.fonts.catalog() are accepted
        "font_name": normalize_font_id(font_name),
    }
    
    status_db[task_id] = {
        "percent": 0,
        "stage": "已成功上传，进入高并发调度队列",
        "status": "queued"
    }
    
    await translation_queue.put(task_metadata)
    
    return {
        "task_id": task_id,
        "status": "queued",
        "message": "Task queued successfully."
    }

@app.delete("/api/v1/translate/cancel/{task_id}")
async def cancel_translation_task(task_id: str):
    """
    Submits a cancellation request for the active translation task.
    """
    if task_id in status_db:
        status = status_db[task_id].get("status")
        if status in ["queued", "processing", "review"]:
            cancelled_tasks.add(task_id)
            status_db[task_id] = {
                "percent": 0,
                "stage": "用户已终止本次翻译任务。",
                "status": "failed",
                "message": "Task cancelled by user."
            }
            return {"status": "cancelled", "message": "Cancellation request submitted."}
    return {"status": "error", "message": "Task not found or already completed."}

class ReviewRequest(BaseModel):
    action: str = "continue"          # "continue" (apply `pages`) | "skip" (render as translated)
    pages: dict[str, list] = {}       # {page: [{id, bbox, text, raw, edited}]}: only the pages the user changed


@app.post("/api/v1/translate/review/{task_id}")
async def review_continue(task_id: str, req: ReviewRequest):
    """Answer a task that is waiting in 'review' (先调框): apply the edits, then render."""
    wait = review_waits.get(task_id)
    if not wait:
        raise HTTPException(status_code=409, detail="这个任务当前没有在等待调框（可能已继续、已取消或已超时）。")
    rows = {}
    if req.action != "skip" and req.pages:
        from core.document_skill import parse_corrections_json, CorrectionsError
        try:
            rows = parse_corrections_json(json.dumps({"pages": req.pages}))
        except CorrectionsError as e:
            raise HTTPException(status_code=400, detail=str(e))
    wait["payload"] = rows
    wait["event"].set()
    return {"status": "resumed", "pages": sorted(rows)}


@app.get("/api/v1/translate/status/{task_id}")
async def get_translation_status_stream(task_id: str):
    """
    SSE stream endpoint pushing real-time workflow statuses and model metrics to client.
    """
    if task_id not in status_db:
        raise HTTPException(status_code=404, detail="Task ID not registered in database.")
        
    async def sse_event_generator():
        last_percent = -1
        review_sent = False
        while True:
            task_status = status_db.get(task_id)
            if not task_status:
                break
                
            percent = task_status.get("percent", 0)
            stage = task_status.get("stage", "")
            status = task_status.get("status", "")
            metrics = task_status.get("metrics", None)
            
            left_review = review_sent and status != "review"
            if left_review:
                review_sent = False
            if percent != last_percent or left_review or status in ["complete", "failed", "review"]:
                last_percent = percent
                
                if status == "complete":
                    yield f"event: complete\ndata: {json.dumps({'download_url': task_status.get('download_url'), 'download_zip_url': task_status.get('download_zip_url'), 'download_xlsx_url': task_status.get('download_xlsx_url'), 'download_json_url': task_status.get('download_json_url'), 'warning': task_status.get('warning', ''), 'edit_data': task_status.get('edit_data'), 'failed_pages': task_status.get('failed_pages', []), 'stage_times': (task_status.get('metrics') or {}).get('stage_times'), 'pipeline_mode': (task_status.get('metrics') or {}).get('pipeline_mode'), 'extract_devices': (task_status.get('metrics') or {}).get('extract_devices'), 'translate_breakdown': (task_status.get('metrics') or {}).get('translate_breakdown'), 'render_breakdown': (task_status.get('metrics') or {}).get('render_breakdown'), 'extract_breakdown': (task_status.get('metrics') or {}).get('extract_breakdown'), 'model_usage': (task_status.get('metrics') or {}).get('model_usage')}, ensure_ascii=False)}\n\n"
                    break
                elif status == "review":
                    if not review_sent:
                        review_sent = True
                        yield f"event: review\ndata: {json.dumps({'stage': stage, 'edit_data': task_status.get('edit_data')}, ensure_ascii=False)}\n\n"
                elif status == "failed":
                    yield f"event: error\ndata: {json.dumps({'message': task_status.get('message', 'Processing pipeline crashed.')})}\n\n"
                    break
                else:
                    yield f"event: progress\ndata: {json.dumps({'percent': percent, 'stage': stage, 'metrics': metrics}, ensure_ascii=False)}\n\n"
            
            await asyncio.sleep(0.5)

    return StreamingResponse(sse_event_generator(), media_type="text/event-stream")

def _remove_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError as e:
        logger.warning(f"Could not remove downloaded file {path}: {e}")


@app.get("/api/v1/download/{filename}")
async def download_translated_file(filename: str):
    """
    Streams file bytes securely for download.
    """
    filepath = os.path.join(STATIC_DIR, filename)
    # Only plain files directly inside STATIC_DIR (the file is deleted after sending, so no "..\\" escapes)
    if os.path.dirname(os.path.realpath(filepath)) != os.path.realpath(STATIC_DIR) or filename.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid file name.")
    if os.path.isfile(filepath):
        media_type = DOWNLOAD_MEDIA_TYPES.get(os.path.splitext(filename)[1].lower(), "application/octet-stream")
        # One-shot download: the file is deleted as soon as it has been sent
        return FileResponse(
            path=filepath,
            filename=filename,
            media_type=media_type,
            background=BackgroundTask(_remove_quietly, filepath)
        )
    else:
        raise HTTPException(status_code=404, detail="Requested file not found on disk.")


if __name__ == "__main__":
    # Ensure port 8000 is clean and free from previous dead processes
    import subprocess
    try:
        res = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
        my_pid = os.getpid()
        for line in res.stdout.splitlines():
            if ':8000' in line and 'LISTENING' in line:
                pid = line.strip().split()[-1]
                if pid.isdigit() and int(pid) != my_pid:
                    os.system(f'taskkill /F /PID {pid} >nul 2>&1')
                    time.sleep(0.5)
    except Exception:
        pass

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
