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
import uuid
import json
import asyncio
import logging
import multiprocessing
import hashlib
import zipfile
import shutil

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


@app.on_event("startup")
async def startup_event():
    global translation_engine
    logger.info("⚡ System Booting... Initializing Pipeline Components...")
    
    # Try GPU speculative decoding, fallbacks gracefully to standard GPU FP16 or CPU Heuristics
    translation_engine = HighPerformanceTranslationEngine(use_gpu=True)
    
    # Start the non-blocking background queue task listener
    asyncio.create_task(translation_worker())
    logger.info("[✓] Background Translation Queue Guardian started successfully!")

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
        force_retranslate = True
        
        status_db[task_id] = {
            "percent": 5,
            "stage": "等待队列调度",
            "status": "processing",
            "metrics": None
        }
        
        try:
            logger.info(f"🚀 Processing Task {task_id} inside background loop...")
            
            # Treat STATIC_DIR as cache: keep only the latest run's files
            try:
                for f in os.listdir(STATIC_DIR):
                    if f.startswith(".git"):
                        continue
                    file_path = os.path.join(STATIC_DIR, f)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                logger.info("Cleared previous files in static directory (cache mode).")
            except Exception as e:
                logger.warning(f"Failed to clear old static files: {e}")
            
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
                for page_num in selected_pages:
                    cache_path = os.path.join(cache_base_dir, f"page_{page_num}.jpg")
                    if os.path.exists(cache_path):
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
            
            import tempfile
            import shutil
            import fitz
            import zipfile
            
            # Create a temp dir for rendered page JPEGs
            temp_dir = tempfile.mkdtemp(prefix="pdf_render_")
            
            renderer = PDFLayoutRenderer(pdf_path, None)
            
            # Semaphore to restrict GPU inpainting to 1 concurrent task to guarantee 4GB VRAM safety
            render_sem = asyncio.Semaphore(1)
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
                
                status_db[task_id]["percent"] = percent
                status_db[task_id]["stage"] = (
                    f"流水线全速并发中: 已提取 {extracted_count}/{total_selected_pages} 页, "
                    f"已翻译 {translated_count}/{total_selected_pages} 页, "
                    f"已重绘 {rendered_count}/{total_selected_pages} 页"
                )
            
            async def render_page_background(page_data, p_num, s_page):
                nonlocal rendered_count
                path = await renderer.render_single_page_to_temp(
                    page_data=page_data,
                    src_page=s_page,
                    translated_text_map=translated_text_map,
                    temp_dir=temp_dir,
                    sem=render_sem
                )
                async with progress_lock:
                    rendered_count += 1
                    update_progress()
                return path

            batch_count = 0
            async for chunk_page in extractor.extract_layout_stream(page_range_list=selected_pages, source_lang=source_lang):
                chunk = [chunk_page]
                layout_data.append(chunk_page)
                
                async with progress_lock:
                    extracted_count += 1
                    update_progress()
                # Check for cancellation
                if task_id in cancelled_tasks:
                    logger.info(f"Task {task_id} cancellation detected. Aborting pipeline.")
                    raise Exception("Task cancelled by user.")
                
                pages_to_translate = []
                for page in chunk:
                    page_num = page["page_num"]
                    cache_path = os.path.join(cache_base_dir, f"page_{page_num}.jpg")
                    
                    if os.path.exists(cache_path):
                        # Page is cached!
                        logger.info(f"Page {page_num} found in cache. Restoring from cache.")
                        temp_dest = os.path.join(temp_dir, f"rendered_page_{page_num}.jpg")
                        await asyncio.to_thread(shutil.copy, cache_path, temp_dest)
                        
                        async with progress_lock:
                            translated_count += 1
                            rendered_count += 1
                            update_progress()
                        
                        fut = asyncio.Future()
                        fut.set_result(temp_dest)
                        render_tasks[page_num] = fut
                    else:
                        pages_to_translate.append(page)
                
                # If there are pages that need translation in this chunk:
                if pages_to_translate:
                    combined_blocks = []
                    # Maps unique_counter -> (page_num, original_block, original_block_id)
                    block_refs = {}
                    unique_counter = 0
                    
                    for page in pages_to_translate:
                        p_num = page["page_num"]
                        for block in page["blocks"]:
                            orig_block_id = block["id"]  # Save real ID BEFORE overwriting
                            block_copy = dict(block)
                            block_copy["id"] = unique_counter
                            block_copy["page_num"] = p_num
                            combined_blocks.append(block_copy)
                            block_refs[unique_counter] = (p_num, block, orig_block_id)
                            unique_counter += 1
                            
                    if combined_blocks:
                        translations, metrics = translation_engine.translate_batch(combined_blocks, source_lang=source_lang)
                        
                        total_metrics["tokens_per_sec"] += metrics["tokens_per_sec"]
                        total_metrics["acceptance_rate"] += metrics["acceptance_rate"]
                        total_metrics["latency_ms"] += metrics["latency_ms"]
                        total_metrics["tokens_generated"] += metrics["tokens_generated"]
                        batch_count += 1
                        
                        for temp_block in combined_blocks:
                            u_id = temp_block["id"]
                            p_num, orig_block, orig_block_id = block_refs[u_id]
                            orig_block["translated_text"] = temp_block.get("translated_text", "")
                            orig_block["is_sfx"] = temp_block.get("is_sfx", False)
                            orig_block["google_trans"] = temp_block.get("google_trans", "")
                            orig_block["cleaned_text"] = temp_block.get("cleaned_text", "")
                            
                            # Key by ORIGINAL block id (what the renderer reads), not unique_counter
                            translated_text_map[(p_num, orig_block_id)] = orig_block["translated_text"]
                            
                    # Start background rendering for translated pages
                    for page in pages_to_translate:
                        p_num = page["page_num"]
                        src_page = src_doc[p_num - 1]
                        
                        async with progress_lock:
                            translated_count += 1
                            update_progress()
                            
                        cache_path = os.path.join(cache_base_dir, f"page_{p_num}.jpg")
                        
                        # Use default args to capture loop variables by value (avoid closure bug)
                        async def render_and_cache(_page_data=page, _p_num=p_num, _s_page=src_page, _c_path=cache_path):
                            dest_path = await render_page_background(_page_data, _p_num, _s_page)
                            if dest_path and os.path.exists(dest_path):
                                await asyncio.to_thread(shutil.copy, dest_path, _c_path)
                            return dest_path
                            
                        render_tasks[p_num] = asyncio.create_task(render_and_cache())
                
                # Yield control briefly to start async tasks
                await asyncio.sleep(0.01)
                
            # Retain original sequential reading order for DOC and JSON generation
            layout_data.sort(key=lambda x: x["page_num"])
                
            if batch_count > 0:
                total_metrics["tokens_per_sec"] /= batch_count
                total_metrics["acceptance_rate"] /= batch_count
                total_metrics["latency_ms"] /= batch_count
                
            # Wait for all scheduled rendering tasks to complete
            temp_paths = {}
            for page_num, task in render_tasks.items():
                temp_paths[page_num] = await task

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
                    if b_raw or b_trans:
                        p_blocks.append({"raw": b_raw, "translated": b_trans})
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
                    json.dump(layout_data, f, ensure_ascii=False, indent=4)
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
                "stage": "全部处理完成！",
                "status": "complete",
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

@app.post("/api/v1/translate/upload")
async def upload_pdf_file(
    file: UploadFile = File(...),
    source_lang: str = Form(...),
    target_lang: str = Form("Simplified Chinese"),
    page_range: str = Form(""),
    force_retranslate: bool = Form(True)
):
    """
    Uploads the raw PDF file, assigns uuid, and queues the task.
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
        "force_retranslate": force_retranslate
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
                    yield f"event: complete\ndata: {json.dumps({'download_url': task_status.get('download_url'), 'download_zip_url': task_status.get('download_zip_url'), 'download_doc_url': task_status.get('download_doc_url'), 'download_json_url': task_status.get('download_json_url')})}\n\n"
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
        media_type = "application/zip" if filename.endswith(".zip") else "application/pdf"
        return FileResponse(
            path=filepath,
            filename=filename,
            media_type=media_type
        )
    else:
        raise HTTPException(status_code=404, detail="Requested file not found on disk.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
