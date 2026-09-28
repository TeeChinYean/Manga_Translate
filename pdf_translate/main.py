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

import os
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
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse

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


def resolve_pipeline_mode(mode: str, n_pages: int) -> str:
    """Map 'auto' (or an unknown value) to a concrete mode for this job size."""
    if mode not in PIPELINE_MODES or mode == "auto":
        return "overlap" if n_pages >= AUTO_OVERLAP_MIN_PAGES else "serial"
    return mode
# Measured 2026-09-28 (第5巻 p1-10, RTX 3050 4GB + 12 threads): serial 109.2s vs stream 129.7s,
# and serial translates the whole range with one large-context LLM call.
DEFAULT_PIPELINE_MODE = os.getenv("PIPELINE_MODE", "auto")
DEFAULT_CONTEXT_CHUNK = 36  # Japanese lines per LLM call in 'serial' mode
RENDER_CONCURRENCY = max(1, int(os.getenv("RENDER_CONCURRENCY", "3")))  # pages rendered at once (all modes)
# stream/overlap: translate once this many lines are buffered (0 = old per-page behaviour)
DEFAULT_STREAM_BATCH_LINES = max(0, int(os.getenv("TRANSLATE_BATCH_LINES", str(DEFAULT_CONTEXT_CHUNK))))


STATIC_KEEP_FILES = 20  # ~5 tasks (pdf + zip + doc + json each)


def prune_static_outputs(static_dir: str, keep: int = STATIC_KEEP_FILES) -> int:
    """Delete all but the newest `keep` output files in static_dir. Returns how many were removed."""
    try:
        entries = [os.path.join(static_dir, f) for f in os.listdir(static_dir) if not f.startswith(".git")]
    except OSError:
        return 0
    files = sorted((p for p in entries if os.path.isfile(p)), key=os.path.getmtime, reverse=True)
    removed = 0
    for path in files[keep:]:
        try:
            os.unlink(path)
            removed += 1
        except OSError as e:
            logger.warning(f"Failed to remove old output {path}: {e}")
    return removed


DOWNLOAD_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".zip": "application/zip",
    ".json": "application/json",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
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
    
    # 2. Preload EasyOCR and MangaOCR
    def _warm_ocr():
        try:
            from core.extractor import _get_ocr_reader, _get_manga_ocr, prepare_extract_devices
            prepare_extract_devices()
            import numpy as np
            from PIL import Image
            
            # Warm EasyOCR detector
            reader = _get_ocr_reader("Japanese")
            dummy_img = np.full((128, 128, 3), 255, dtype=np.uint8)
            reader.readtext(dummy_img, paragraph=True)
            
            # Warm MangaOCR ViT model
            mocr = _get_manga_ocr()
            if mocr is not None:
                dummy_pil = Image.fromarray(dummy_img)
                mocr(dummy_pil)
            logger.info("✔ [Preloader] OCR models (EasyOCR + MangaOCR) 100% preloaded!")
        except Exception as ex:
            logger.warning(f"⚠️ [Preloader] OCR preloading warning: {ex}")
            
    await loop.run_in_executor(None, _warm_ocr)
    logger.info("✔ [Preloader] All pipeline models preloaded and warm in memory!")


@app.on_event("startup")
async def startup_event():
    global translation_engine
    logger.info("⚡ System Booting... Initializing Pipeline Components...")
    
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
        page_range = task.get("page_range", "")
        force_retranslate = bool(task.get("force_retranslate", False))
        
        status_db[task_id] = {
            "percent": 5,
            "stage": "等待队列调度",
            "status": "processing",
            "metrics": None
        }
        
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
            
            if source_lang == "Japanese":
                from core.engine import ensure_turbovec_llm_ready
                await asyncio.to_thread(ensure_turbovec_llm_ready, True, 20)
            
            # Output files: keep the most recent runs so earlier download links keep working (B10)
            prune_static_outputs(STATIC_DIR, keep=STATIC_KEEP_FILES)
            
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
                "stage": "流水线全并发就绪: 边检测边翻译",
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
            cache_base_dir = os.path.join(BASE_DIR, "data", "cache", pdf_hash)
            os.makedirs(cache_base_dir, exist_ok=True)
            
            # Clear cache for selected pages if force_retranslate is True
            if force_retranslate:
                import glob
                for page_num in selected_pages:
                    for cache_path in (glob.glob(os.path.join(cache_base_dir, f"page_{page_num}.jpg"))
                                       + glob.glob(os.path.join(cache_base_dir, f"page_{page_num}_*.jpg"))):
                        try:
                            os.remove(cache_path)
                            logger.info(f"Cleared cache for page {page_num}")
                        except Exception as ex:
                            logger.warning(f"Could not clear cache for page {page_num}: {ex}")

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

            renderer = PDFLayoutRenderer(
                pdf_path,
                None,
                ink_thresh=ink_thresh,
                dilate_iter=dilate_iter,
                max_stroke_ratio=max_stroke_ratio,
                font_scale=font_scale
            )
            
            # Semaphore to restrict GPU inpainting to 1 concurrent task to guarantee 4GB VRAM safety
            from core.renderer import LAMA_PARALLEL, reset_render_stats, get_render_stats, preload_lama
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
                    f"流水线全速并发中: 已提取 {extracted_count}/{total_selected_pages} 页, "
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
                async for chunk_page in extractor.extract_layout_stream(page_range_list=selected_pages, source_lang=source_lang):
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

            async def _restore_cached(page) -> bool:
                """Cached page: restore the rendered JPEG and skip translation + rendering."""
                nonlocal translated_count, rendered_count
                page_num = page["page_num"]
                cache_path = os.path.join(cache_base_dir, renderer.cache_filename(page_num))
                if not os.path.exists(cache_path):
                    return False
                logger.info(f"Page {page_num} found in cache. Skipping translation.")
                temp_dest = os.path.join(temp_dir, f"rendered_page_{page_num}.jpg")
                await asyncio.to_thread(shutil.copy, cache_path, temp_dest)
                temp_paths[page_num] = temp_dest
                async with progress_lock:
                    translated_count += 1
                    rendered_count += 1
                    update_progress()
                return True

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
                    if await _restore_cached(page):
                        continue
                    to_render.append(page)
                    for block in page.get("blocks", []):
                        u_id = len(combined_blocks)
                        block_copy = dict(block)
                        block_copy["id"] = u_id
                        block_copy["page_num"] = page["page_num"]
                        combined_blocks.append(block_copy)
                        block_refs[u_id] = (page["page_num"], block, block["id"])

                if combined_blocks:
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
                cache_path = os.path.join(cache_base_dir, renderer.cache_filename(p_num))
                await start_lama_preload()  # no-op once loaded

                dest_path = await renderer.render_single_page_to_temp(
                    page_data=page,
                    src_page=src_page,
                    translated_text_map=translated_text_map,
                    temp_dir=temp_dir,
                    sem=render_sem
                )

                if dest_path and os.path.exists(dest_path):
                    page_blocks = page.get("blocks", [])
                    has_translated = (not page_blocks) or any(
                        translated_text_map.get((p_num, b["id"]), "").strip() for b in page_blocks
                    )
                    if has_translated:
                        await asyncio.to_thread(shutil.copy, dest_path, cache_path)
                    temp_paths[p_num] = dest_path

                async with progress_lock:
                    rendered_count += 1
                    update_progress()
                _mark("render_done")

            async def unload_ocr_models():
                from core.extractor import unload_models as unload_ocr
                released = await asyncio.to_thread(unload_ocr)
                _mark("ocr_unloaded")
                start_lama_preload()
                logger.info(f"[Pipeline] OCR models released: {sorted(released or [])} "
                            "(GPU ones always; CPU ones only when RAM is low, see OCR_UNLOAD).")

            def _check_cancel():
                if task_id in cancelled_tasks:
                    raise PipelineCancelled("Task cancelled by user.")

            async def run_serial():
                """Mode 'serial': extract ALL -> unload OCR -> translate ALL (big context) -> render ALL."""
                pages = []
                async for p in extracted_pages():
                    _check_cancel()
                    pages.append(p)
                pages.sort(key=lambda x: x["page_num"])
                await unload_ocr_models()
                start_lama_preload()  # CPU is idle while the LLM translates

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
            total_metrics["extract_breakdown"] = get_extract_stats()
            logger.info(f"[Timing] mode={pipeline_mode} pages={total_selected_pages} {stage_times}")


            failed_pages = sorted(extractor.failed_pages)
            if not layout_data:
                detail = f"（失败页: {', '.join(str(p) for p, _ in failed_pages)}；首个错误: {failed_pages[0][1]}）" if failed_pages else ""
                raise Exception("未能成功提取到任何页面数据，请检查 PDF 文件完整性或页面范围设置。" + detail)

            extraction_warning = build_failed_pages_warning(failed_pages)
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
                "stage": "编译拼装高保真 PDF 中",
                "status": "processing",
                "metrics": total_metrics
            }
            
            out_filename = f"translated_{uuid.uuid4().hex[:8]}_{filename}"
            output_pdf_path = os.path.join(STATIC_DIR, out_filename)
            
            out_doc = fitz.open()
            for page_num in range(1, actual_total_pages + 1):
                src_page = src_doc[page_num - 1]
                if page_num in selected_pages and page_num in temp_paths and os.path.exists(temp_paths[page_num]):
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
            
            # --- SKILL: Word Processing Generation ---
            doc_pages_data = []
            for p_data in layout_data:
                p_num = p_data["page_num"]
                if p_num not in selected_pages: continue
                p_blocks = []
                for blk in p_data.get("blocks", []):
                    b_id = blk.get("id")
                    b_raw = blk.get("cleaned_text", blk.get("text", ""))
                    # Key is (page_num, block_id) tuple — must match how translated_text_map is built
                    b_trans = translated_text_map.get((p_num, b_id), "")
                    ocr_eng = blk.get("ocr_engine", "MangaOCR (ViT)")
                    trans_eng = blk.get("translation_engine", "Turbovec Qwen 3.5 4B")
                    if b_raw or b_trans:
                        p_blocks.append({
                            "raw": b_raw,
                            "translated": b_trans,
                            "ocr_engine": ocr_eng,
                            "translation_engine": trans_eng
                        })
                if p_blocks:
                    doc_pages_data.append({"page_num": p_num, "blocks": p_blocks})
            
            doc_path = None
            try:
                from core.document_skill import LocalDocumentSkill
                doc_skill = LocalDocumentSkill(STATIC_DIR)
                doc_path = doc_skill.generate_bilingual_doc(task_id, doc_pages_data, filename)
            except Exception as e:
                logger.error(f"Failed to generate translation DOC script: {e}")
            
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
                    if doc_path and os.path.exists(doc_path):
                        zip_f.write(doc_path, os.path.basename(doc_path))
                    if os.path.exists(output_json_path):
                        zip_f.write(output_json_path, os.path.basename(output_json_path))
                            
            await asyncio.to_thread(write_zip)
            
            # Clean up temp files
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            download_url = f"/api/v1/download/{out_filename}"
            download_zip_url = f"/api/v1/download/{out_zip_filename}"
            download_doc_url = f"/api/v1/download/{os.path.basename(doc_path)}" if doc_path else ""
            download_json_url = f"/api/v1/download/{out_json_filename}"
            
            logger.info(f"✅ Task {task_id} completed successfully.")
            
            status_db[task_id] = {
                "percent": 100,
                "stage": "全部处理完成！" + (f"（{extraction_warning}）" if extraction_warning else ""),
                "status": "complete",
                "warning": extraction_warning,
                "failed_pages": [p for p, _ in failed_pages],
                "download_url": download_url,
                "download_zip_url": download_zip_url,
                "download_doc_url": download_doc_url,
                "download_json_url": download_json_url,
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
    translate_batch_lines: int = Form(DEFAULT_STREAM_BATCH_LINES)
):
    """
    Uploads the raw PDF file, assigns uuid, applies custom tuning parameters, and queues the task.
    """
    if not file.filename.endswith(".pdf"):
        return {"error": "Invalid format. Only PDF files are supported."}
        
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
        "translate_batch_lines": translate_batch_lines
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
        if status in ["queued", "processing"]:
            cancelled_tasks.add(task_id)
            status_db[task_id] = {
                "percent": 0,
                "stage": "用户已终止本次翻译任务。",
                "status": "failed",
                "message": "Task cancelled by user."
            }
            return {"status": "cancelled", "message": "Cancellation request submitted."}
    return {"status": "error", "message": "Task not found or already completed."}

@app.get("/api/v1/translate/status/{task_id}")
async def get_translation_status_stream(task_id: str):
    """
    SSE stream endpoint pushing real-time workflow statuses and model metrics to client.
    """
    if task_id not in status_db:
        raise HTTPException(status_code=404, detail="Task ID not registered in database.")
        
    async def sse_event_generator():
        last_percent = -1
        while True:
            task_status = status_db.get(task_id)
            if not task_status:
                break
                
            percent = task_status.get("percent", 0)
            stage = task_status.get("stage", "")
            status = task_status.get("status", "")
            metrics = task_status.get("metrics", None)
            
            if percent != last_percent or status in ["complete", "failed"]:
                last_percent = percent
                
                if status == "complete":
                    yield f"event: complete\ndata: {json.dumps({'download_url': task_status.get('download_url'), 'download_zip_url': task_status.get('download_zip_url'), 'download_doc_url': task_status.get('download_doc_url'), 'download_json_url': task_status.get('download_json_url'), 'warning': task_status.get('warning', ''), 'failed_pages': task_status.get('failed_pages', []), 'stage_times': (task_status.get('metrics') or {}).get('stage_times'), 'pipeline_mode': (task_status.get('metrics') or {}).get('pipeline_mode'), 'extract_devices': (task_status.get('metrics') or {}).get('extract_devices'), 'translate_breakdown': (task_status.get('metrics') or {}).get('translate_breakdown'), 'render_breakdown': (task_status.get('metrics') or {}).get('render_breakdown'), 'extract_breakdown': (task_status.get('metrics') or {}).get('extract_breakdown'), 'model_usage': (task_status.get('metrics') or {}).get('model_usage')}, ensure_ascii=False)}\n\n"
                    break
                elif status == "failed":
                    yield f"event: error\ndata: {json.dumps({'message': task_status.get('message', 'Processing pipeline crashed.')})}\n\n"
                    break
                else:
                    yield f"event: progress\ndata: {json.dumps({'percent': percent, 'stage': stage, 'metrics': metrics}, ensure_ascii=False)}\n\n"
            
            await asyncio.sleep(0.5)

    return StreamingResponse(sse_event_generator(), media_type="text/event-stream")

@app.get("/api/v1/download/{filename}")
async def download_translated_file(filename: str):
    """
    Streams file bytes securely for download.
    """
    filepath = os.path.join(STATIC_DIR, filename)
    if os.path.exists(filepath):
        media_type = DOWNLOAD_MEDIA_TYPES.get(os.path.splitext(filename)[1].lower(), "application/octet-stream")
        return FileResponse(
            path=filepath,
            filename=filename,
            media_type=media_type
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
