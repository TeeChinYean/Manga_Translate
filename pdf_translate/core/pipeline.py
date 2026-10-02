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

async def _run_pipeline_async(pdf_path, page_range_str, source_lang, target_lang, task_id, progress_callback,
                              ink_thresh: int = 95, dilate_iter: int = 2,
                              max_stroke_ratio: float = 0.35, font_scale: float = 1.0):
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

    temp_dir = cache_base_dir = None
    try:
        _safe_callback(10, "C# 正在内存级融合 PyTorch...", "processing")
        
        # 1. Initialize models
        extractor = PDFLayoutExtractor(pdf_path)
        from core.extractor import prepare_extract_devices
        prepare_extract_devices()
        translation_engine = HighPerformanceTranslationEngine()
        
        output_pdf_path = pdf_path.replace('.pdf', '_translated.pdf')
        renderer = PDFLayoutRenderer(
            pdf_path,
            output_pdf_path,
            ink_thresh=ink_thresh,
            dilate_iter=dilate_iter,
            max_stroke_ratio=max_stroke_ratio,
            font_scale=font_scale
        )
        
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
            nonlocal translated_count, text_cache_dirty

            # 整批页面合并成一个块列表，再按 CTX_SIZE 句一组喂给 LLM（维持原版上下文大小，避免质量下降）
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
                # ── 文本缓存命中检查：命中则跳过 LLM ──
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

                CTX_SIZE = 12
                for i in range(0, len(cache_miss_blocks), CTX_SIZE):
                    if task_id in cancelled_tasks:
                        raise Exception("Task cancelled by user.")
                    sub_batch = cache_miss_blocks[i:i + CTX_SIZE]
                    await asyncio.to_thread(
                        translation_engine.translate_batch,
                        sub_batch,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        context_chunk_size=CTX_SIZE,
                    )

                # ── 新译文写入缓存 ──
                for blk in cache_miss_blocks:
                    raw = blk.get("text", "").strip()
                    translated = (blk.get("translated_text") or "").strip()
                    if raw and translated:
                        text_cache[_text_cache_key(raw)] = translated
                        text_cache_dirty = True

                # 同步回原页面块，并写入 translated_text_map
                for temp_block in combined_blocks:
                    p_num_ref, orig_block, orig_block_id = block_refs[temp_block["id"]]
                    orig_block["translated_text"] = temp_block.get("translated_text", "")
                    orig_block["is_sfx"] = temp_block.get("is_sfx", False)
                    orig_block["google_trans"] = temp_block.get("google_trans", "")
                    orig_block["cleaned_text"] = temp_block.get("cleaned_text", "")
                    translated_text_map[(p_num_ref, orig_block_id)] = orig_block["translated_text"]

            async with progress_lock:
                translated_count += len(chunk)
                update_progress()

        # 自动选择：CUDA 下显存充足就不强制卸载模型，节省重新载入时间
        import torch
        from core.gpu_budget import free_vram_mb

        def should_unload():
            if not torch.cuda.is_available():
                return True
            free, _ = free_vram_mb()
            # LaMa 与 LLM 需要较多显存，剩余不足 2GB 时清理
            return free is not None and free < 2048

        async def render_single_page_job(page):
            nonlocal rendered_count
            p_num = page["page_num"]
            cache_path = os.path.join(cache_base_dir, renderer.cache_filename(p_num))

            if os.path.exists(cache_path):
                temp_dest = os.path.join(temp_dir, f"rendered_page_{p_num}.jpg")
                await asyncio.to_thread(shutil.copy, cache_path, temp_dest)
                render_tasks[p_num] = temp_dest
            else:
                src_page = doc[p_num - 1]
                dest_path = await render_page_background(page, p_num, src_page)
                page_blocks = page.get("blocks", [])
                has_translated = (not page_blocks) or any(
                    translated_text_map.get((p_num, b["id"]), "").strip() for b in page_blocks)
                if dest_path and os.path.exists(dest_path) and has_translated:
                    await asyncio.to_thread(shutil.copy, dest_path, cache_path)
                render_tasks[p_num] = dest_path

            async with progress_lock:
                rendered_count += 1
                update_progress()

        translation_errors = []

        async def process_batch(chunk):
            print(f"\n[Pipeline] 🔄 处理批次: {len(chunk)} 页")
            # 翻译：全部翻译完才启动 LaMa，避免 LLM 与 LaMa 抢 GPU
            try:
                await process_page_chunk_translate(chunk)
            except Exception as e:
                if task_id in cancelled_tasks:
                    raise
                print(f"\n[Pipeline] 🚨 翻译阶段异常: {e}")
                translation_errors.append(e)
            if translation_errors:
                raise Exception(f"翻译阶段出错（{len(translation_errors)} 批）: {translation_errors[0]}")

            # OCR 已全部完成，重绘前视显存情况释放 OCR 模型
            if should_unload():
                from core.extractor import unload_models as unload_ocr
                await asyncio.to_thread(unload_ocr)
                if torch.cuda.is_available() and torch.cuda.is_initialized():
                    torch.cuda.empty_cache()

            # 重绘
            await asyncio.gather(*[render_single_page_job(page) for page in chunk])

        # 用户要求：等所有选定页全部提取并翻译完后，再集中启动 LaMa（不再分批）
        layout_data = []
        current_chunk = []

        async for chunk_page in extractor.extract_layout_stream(page_range_list=selected_pages, source_lang=source_lang):
            if task_id in cancelled_tasks:
                raise Exception("Task cancelled by user.")

            layout_data.append(chunk_page)
            current_chunk.append(chunk_page)

            async with progress_lock:
                extracted_count += 1
                update_progress()

        if current_chunk:
            await process_batch(current_chunk)

        if layout_data:
            print(f"\n[Pipeline] 🧹 所有页面处理完毕，释放重绘资源...")
            from core.renderer import unload_models as unload_renderer
            await asyncio.to_thread(unload_renderer)
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.empty_cache()

        rendered_images = render_tasks
            
        _safe_callback(95, "正在将所有画面合并为最终 PDF...", "processing")
        
        # Assemble final PDF in order
        out_doc = fitz.open()
        for page_num in range(1, total_doc_pages + 1):
            src_page = doc[page_num - 1]
            rendered = rendered_images.get(page_num)
            if rendered and os.path.exists(rendered):
                out_page = out_doc.new_page(
                    width=src_page.rect.width,
                    height=src_page.rect.height
                )
                out_page.insert_image(out_page.rect, filename=rendered)
            else:
                # Not selected / not rendered: keep the original page instead of a blank one (B9)
                out_doc.insert_pdf(doc, from_page=page_num - 1, to_page=page_num - 1)
                
        doc.close()
        out_doc.save(output_pdf_path, garbage=4, deflate=True)
        out_doc.close()
        
        # 写入文本翻译缓存（只有新翻译才写）
        if text_cache_dirty:
            _save_text_cache(cache_base_dir, text_cache)
            print(f"[Pipeline] 💾 翻译缓存已更新 ({len(text_cache)} 条)")

        failed = sorted(extractor.failed_pages)
        if failed:
            nums = ", ".join(str(p) for p, _ in failed)
            _safe_callback(100, f"翻译完成（{len(failed)} 页提取失败，未翻译: 第 {nums} 页）", "complete")
        else:
            _safe_callback(100, "翻译完成", "complete")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        progress_callback(100, f"执行崩溃: {str(e)}", "failed")
    finally:
        # Rendered JPEGs and the page/text cache never outlive the task
        for d in (temp_dir, cache_base_dir):
            if d:
                shutil.rmtree(d, ignore_errors=True)
        # Unload models and clear VRAM cache
        try:
            from core.renderer import unload_models as unload_renderer
            unload_renderer()
            from core.extractor import unload_models as unload_ocr
            unload_ocr()
            import torch
            if torch.cuda.is_available() and torch.cuda.is_initialized():  # never create a CUDA context here (B29)
                torch.cuda.empty_cache()
        except Exception:
            pass

def run_pipeline(pdf_path, page_range_str, source_lang, target_lang, task_id, progress_callback,
                 ink_thresh: int = 95, dilate_iter: int = 2,
                 max_stroke_ratio: float = 0.30, font_scale: float = 1.0):
    """
    Synchronous entrypoint called from C# to run the async pipeline.
    """
    asyncio.run(_run_pipeline_async(pdf_path, page_range_str, source_lang, target_lang, task_id, progress_callback,
                                   ink_thresh=ink_thresh, dilate_iter=dilate_iter,
                                   max_stroke_ratio=max_stroke_ratio, font_scale=font_scale))
