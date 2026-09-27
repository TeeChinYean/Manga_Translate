import os
import json
import time
import asyncio
import hashlib
import shutil
import tempfile
import fitz
from PIL import Image
from core.extractor import PDFLayoutExtractor
from core.engine import HighPerformanceTranslationEngine
from core.renderer import PDFLayoutRenderer

# Keep track of cancelled tasks
cancelled_tasks = set()

def cancel_task(task_id):
    cancelled_tasks.add(task_id)


def _text_cache_key(text: str) -> str:
    """将原文生成短哈希作为缓存 key"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def _load_text_cache(cache_base_dir: str) -> dict:
    """加载文本翻译缓存，key=原文hash, value=译文"""
    path = os.path.join(cache_base_dir, "translations.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_text_cache(cache_base_dir: str, cache: dict):
    """持久化文本翻译缓存"""
    path = os.path.join(cache_base_dir, "translations.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Pipeline] ⚠️  写入翻译缓存失败: {e}")

async def _run_pipeline_async(pdf_path, page_range_str, source_lang, target_lang, task_id, progress_callback):
    def _safe_callback(percent: int, message: str, status: str):
        """
        Safe wrapper for the C# progress_callback delegate.
        If dotnet has already shut down, the delegate becomes a dangling pointer
        and calling it raises System.NullReferenceException in Python.Runtime.
        We catch all exceptions here to prevent Python from crashing.
        """
        try:
            progress_callback(int(percent), str(message), str(status))
        except Exception as cb_err:
            print(f"[Pipeline] ⚠️  progress_callback 已失效 (C# 可能已关闭): {cb_err}")

    try:
        _safe_callback(10, "C# 正在内存级融合 PyTorch...", "processing")
        
        # 1. Initialize models
        extractor = PDFLayoutExtractor(pdf_path)
        translation_engine = HighPerformanceTranslationEngine()
        
        output_pdf_path = pdf_path.replace('.pdf', '_translated.pdf')
        renderer = PDFLayoutRenderer(pdf_path, output_pdf_path)
        
        # 2. Parse page range
        doc = fitz.open(pdf_path)
        total_doc_pages = len(doc)
        
        selected_pages = []
        if page_range_str and page_range_str.strip():
            for part in page_range_str.split(','):
                part = part.strip()
                if not part: continue
                if '-' in part:
                    rng = part.split('-')
                    if len(rng) == 2:
                        try:
                            start = max(1, int(rng[0]))
                            end = min(total_doc_pages, int(rng[1]))
                            selected_pages.extend(range(start, end + 1))
                        except ValueError:
                            pass
                else:
                    try:
                        p = int(part)
                        if 1 <= p <= total_doc_pages:
                            selected_pages.append(p)
                    except ValueError:
                        pass
            selected_pages = sorted(list(set(selected_pages)))
            
        if not selected_pages:
            selected_pages = list(range(1, total_doc_pages + 1))
            
        total_selected_pages = len(selected_pages)
        if total_selected_pages == 0:
            doc.close()
            progress_callback(100, "翻译失败：没有选择有效的页码", "failed")
            return
            
        # 3. Calculate MD5 for cache
        hasher = hashlib.md5()
        with open(pdf_path, 'rb') as f:
            for chunk in iter(lambda: f.read(1024*1024), b''):
                hasher.update(chunk)
        pdf_hash = hasher.hexdigest()
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_base_dir = os.path.join(base_dir, "data/cache", pdf_hash)
        os.makedirs(cache_base_dir, exist_ok=True)
        
        # 加载文本翻译缓存（跨会话持久化，避免重复调用 LLM）
        text_cache = _load_text_cache(cache_base_dir)
        text_cache_dirty = False  # 标记是否有新翻译需要写入
        
        _safe_callback(15, "流水线全并发就绪: 边检测边翻译 (Python 驱动)", "processing")
        
        extracted_count = 0
        translated_count = 0
        rendered_count = 0
        progress_lock = asyncio.Lock()
        
        def update_progress():
            ex_part = int(15.0 * extracted_count / max(1, total_selected_pages))
            trans_part = int(35.0 * translated_count / max(1, total_selected_pages))
            render_part = int(30.0 * rendered_count / max(1, total_selected_pages))
            percent = 15 + ex_part + trans_part + render_part
            _safe_callback(
                percent,
                f"流水线全速并发中: 已提取 {extracted_count}/{total_selected_pages} 页, "
                f"已翻译 {translated_count}/{total_selected_pages} 页, "
                f"已重绘 {rendered_count}/{total_selected_pages} 页",
                "processing"
            )
            
        temp_dir = tempfile.mkdtemp(prefix="pdf_render_")
        render_sem = asyncio.Semaphore(3)
        render_tasks = {}
        translated_text_map = {}
        
        async def render_page_background(page_data, p_num, s_page):
            path = await renderer.render_single_page_to_temp(
                page_data=page_data,
                src_page=s_page,
                translated_text_map=translated_text_map,
                temp_dir=temp_dir,
                sem=render_sem
            )
            return path
            
        async def process_page_chunk_translate(chunk):
            nonlocal translated_count
            # 所有页面都经过完整翻译流程，不再以 jpg 缓存判断是否跳过
            # jpg 缓存只在 Stage 2 用于跳过重绘（代表上次运行已完整成功重绘）
            combined_blocks = []
            block_refs = {}
            unique_counter = 0
            
            for page in chunk:
                p_num = page["page_num"]
                for block in page["blocks"]:
                    orig_block_id = block["id"]
                    block_copy = dict(block)
                    block_copy["id"] = unique_counter
                    block_copy["page_num"] = p_num
                    combined_blocks.append(block_copy)
                    block_refs[unique_counter] = (p_num, block, orig_block_id)
                    unique_counter += 1
                        
            if combined_blocks:
                # ── 文本缓存命中检查：直接跳过 LLM 调用 ──────────────────
                cache_miss_blocks = []
                for blk in combined_blocks:
                    raw = blk.get("text", "").strip()
                    ck = _text_cache_key(raw) if raw else None
                    if ck and ck in text_cache:
                        blk["translated_text"] = text_cache[ck]
                        blk["_from_cache"] = True
                    else:
                        cache_miss_blocks.append(blk)
                
                cache_hits = len(combined_blocks) - len(cache_miss_blocks)
                if cache_hits > 0:
                    print(f"[Pipeline] ⚡ 文本缓存命中 {cache_hits}/{len(combined_blocks)} 个块，直接跳过 LLM")
                
                if cache_miss_blocks:
                    print(f"\n[Pipeline] 👉 正在翻译 {len(cache_miss_blocks)}/{len(combined_blocks)} 个未命中缓存的文本块...")
                    await asyncio.to_thread(translation_engine.translate_batch, cache_miss_blocks, source_lang=source_lang)
                    
                    # ── 翻译完成后写入缓存 ──────────────────────────────
                    nonlocal text_cache_dirty
                    for blk in cache_miss_blocks:
                        raw = blk.get("text", "").strip()
                        translated = blk.get("translated_text", "").strip()
                        if raw and translated:
                            ck = _text_cache_key(raw)
                            text_cache[ck] = translated
                            text_cache_dirty = True
                
                # 把 combined_blocks 的翻译结果同步回各页面块，并写入 translated_text_map
                for temp_block in combined_blocks:
                    u_id = temp_block["id"]
                    p_num_ref, orig_block, orig_block_id = block_refs[u_id]
                    orig_block["translated_text"] = temp_block.get("translated_text", "")
                    orig_block["is_sfx"] = temp_block.get("is_sfx", False)
                    orig_block["google_trans"] = temp_block.get("google_trans", "")
                    orig_block["cleaned_text"] = temp_block.get("cleaned_text", "")
                    translated_text_map[(p_num_ref, orig_block_id)] = orig_block["translated_text"]
                    
            async with progress_lock:
                translated_count += len(chunk)
                update_progress()
                        
            await asyncio.sleep(0.01)

        # ── 阶段 1：先提取与翻译同时进行 (Stage 1: Streaming Extraction + Concurrent Translation) ──
        # 提取(EasyOCR on CPU + MangaOCR) 与 翻译(LLM) 边提取边翻译；重绘暂不启动，全部算力与内存优先供给提取与翻译
        layout_data = []
        current_chunk = []
        PAGES_PER_BATCH = 1
        
        translation_queue = asyncio.Queue()
        
        async def translation_worker():
            while True:
                chunk = await translation_queue.get()
                if chunk is None:
                    translation_queue.task_done()
                    break
                try:
                    await process_page_chunk_translate(chunk)
                except Exception as e:
                    print(f"\n[Pipeline] 🚨 后台翻译线程异常: {e}")
                finally:
                    translation_queue.task_done()

        worker_task = asyncio.create_task(translation_worker())
        
        async for chunk_page in extractor.extract_layout_stream(page_range_list=selected_pages, source_lang=source_lang):
            if task_id in cancelled_tasks:
                await translation_queue.put(None)
                raise Exception("Task cancelled by user.")
                
            layout_data.append(chunk_page)
            current_chunk.append(chunk_page)
            
            async with progress_lock:
                extracted_count += 1
                update_progress()
                
            if len(current_chunk) >= PAGES_PER_BATCH:
                await translation_queue.put(current_chunk)
                current_chunk = []
                
        if current_chunk:
            await translation_queue.put(current_chunk)
            
        # ── 阶段 2：OCR 完成即刻彻底卸载释放 (Stage 2: Unload OCR models immediately) ──
        print(f"\n[Pipeline] 🧹 所有页面 OCR 提取完毕，立即彻底释放 EasyOCR 与 MangaOCR (RAM 与显存)...")
        from core.extractor import unload_models as unload_ocr
        await asyncio.to_thread(unload_ocr)
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 等待后台翻译队列全部完成收尾
        await translation_queue.put(None)
        await worker_task
        print(f"\n[Pipeline] 🧹 翻译阶段已全部完成，已准备好所有页面的翻译文本映射！")

        # ── 阶段 3：全面启动 OpenCV Telea 极速重绘与排版 (Stage 3: OpenCV Telea Inpainting & Layout) ──
        _safe_callback(75, "OCR完成且模型已彻底卸载，全面启动 OpenCV Telea 极速重绘与排版...", "processing")
        print(f"[Pipeline] 🎨 开始使用 OpenCV Telea 极速重绘 (共 {len(layout_data)} 页)...")
        
        render_sem = asyncio.Semaphore(4)
        
        async def render_single_page_job(page):
            import shutil
            nonlocal rendered_count
            p_num = page["page_num"]
            cache_path = os.path.join(cache_base_dir, f"page_{p_num}.jpg")
            
            if os.path.exists(cache_path):
                temp_dest = os.path.join(temp_dir, f"rendered_page_{p_num}.jpg")
                await asyncio.to_thread(shutil.copy, cache_path, temp_dest)
                render_tasks[p_num] = temp_dest
            else:
                src_page = doc[p_num - 1]
                dest_path = await render_page_background(page, p_num, src_page)
                if dest_path and os.path.exists(dest_path):
                    await asyncio.to_thread(shutil.copy, dest_path, cache_path)
                render_tasks[p_num] = dest_path
                
            async with progress_lock:
                rendered_count += 1
                update_progress()

        await asyncio.gather(*[render_single_page_job(page) for page in layout_data])
        print(f"[Pipeline] 🎨 所有页面 OpenCV Telea 重绘与排版全部完成！")

        if layout_data:
            print(f"\n[Pipeline] 🧹 释放重绘资源...")
            from core.renderer import unload_models as unload_renderer
            await asyncio.to_thread(unload_renderer)
            
        # 组装图片
        rendered_images = render_tasks
            
        _safe_callback(95, "正在将所有画面合并为最终 PDF...", "processing")
        
        # Assemble final PDF in order
        out_doc = fitz.open()
        for page_num in range(1, total_doc_pages + 1):
            src_page = doc[page_num - 1]
            out_page = out_doc.new_page(
                width=src_page.rect.width,
                height=src_page.rect.height
            )
            if page_num in rendered_images and os.path.exists(rendered_images[page_num]):
                out_page.insert_image(out_page.rect, filename=rendered_images[page_num])
                
        doc.close()
        out_doc.save(output_pdf_path, garbage=4, deflate=True)
        out_doc.close()
        
        # 写入文本翻译缓存（只有新翻译才写）
        if text_cache_dirty:
            _save_text_cache(cache_base_dir, text_cache)
            print(f"[Pipeline] 💾 翻译缓存已更新 ({len(text_cache)} 条)")
        
        # Clean up temp dir
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass
            
        _safe_callback(100, "翻译完成", "complete")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        progress_callback(100, f"执行崩溃: {str(e)}", "failed")
    finally:
        # Unload models and clear VRAM cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

def run_pipeline(pdf_path, page_range_str, source_lang, target_lang, task_id, progress_callback):
    """
    Synchronous entrypoint called from C# to run the async pipeline.
    """
    asyncio.run(_run_pipeline_async(pdf_path, page_range_str, source_lang, target_lang, task_id, progress_callback))
