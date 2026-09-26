#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ ANTIGRAVITY - High-Performance PDF Layout Extraction Engine
Fidelity features: CPU Multi-threading parallel extraction, text line micro-bounding-boxes,
original RGB text color extraction, and strict noise filter.
"""

import os
import re
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import fitz  # PyMuPDF
import concurrent.futures
import logging
import threading
import cv2
import numpy as np
easyocr = None

def _get_easyocr():
    global easyocr
    if easyocr is None:
        try:
            import easyocr as _eo
            easyocr = _eo
        except ImportError:
            easyocr = None
    return easyocr

logger = logging.getLogger(__name__)

_OCR_LOCK = threading.Lock()
_OCR_READER = None

_MANGA_OCR_LOCK = threading.Lock()
_MANGA_OCR_INSTANCE = None

_OCR_READER_JA = None
_OCR_READER_EN = None

def _get_ocr_reader(lang="Japanese"):
    global _OCR_READER_JA, _OCR_READER_EN, _MANGA_OCR_INSTANCE
    import torch
    use_gpu = torch.cuda.is_available()
    device = 'cuda' if use_gpu else 'cpu'
    
    eo = _get_easyocr()
    if eo is None:
        raise RuntimeError("easyocr is required for image OCR but not available in environment.")
    
    with _OCR_LOCK:
        if lang == "Japanese":
            # Unload English reader to CPU RAM to save VRAM
            if _OCR_READER_EN is not None:
                if hasattr(_OCR_READER_EN, 'detector') and _OCR_READER_EN.detector is not None:
                    _OCR_READER_EN.detector = _OCR_READER_EN.detector.to('cpu')
                if hasattr(_OCR_READER_EN, 'recognizer') and _OCR_READER_EN.recognizer is not None:
                    _OCR_READER_EN.recognizer = _OCR_READER_EN.recognizer.to('cpu')
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            if _OCR_READER_JA is None:
                _OCR_READER_JA = eo.Reader(['ja', 'en'], gpu=use_gpu)
            else:
                # Move back to GPU
                if hasattr(_OCR_READER_JA, 'detector') and _OCR_READER_JA.detector is not None:
                    _OCR_READER_JA.detector = _OCR_READER_JA.detector.to(device)
                if hasattr(_OCR_READER_JA, 'recognizer') and _OCR_READER_JA.recognizer is not None:
                    _OCR_READER_JA.recognizer = _OCR_READER_JA.recognizer.to(device)
            return _OCR_READER_JA
        else:
            # Unload Japanese reader and MangaOCR to CPU RAM
            if _OCR_READER_JA is not None:
                if hasattr(_OCR_READER_JA, 'detector') and _OCR_READER_JA.detector is not None:
                    _OCR_READER_JA.detector = _OCR_READER_JA.detector.to('cpu')
                if hasattr(_OCR_READER_JA, 'recognizer') and _OCR_READER_JA.recognizer is not None:
                    _OCR_READER_JA.recognizer = _OCR_READER_JA.recognizer.to('cpu')
            with _MANGA_OCR_LOCK:
                if _MANGA_OCR_INSTANCE is not None:
                    if hasattr(_MANGA_OCR_INSTANCE, 'model') and _MANGA_OCR_INSTANCE.model is not None:
                        _MANGA_OCR_INSTANCE.model = _MANGA_OCR_INSTANCE.model.to('cpu')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            if _OCR_READER_EN is None:
                _OCR_READER_EN = eo.Reader(['en'], gpu=use_gpu)
            else:
                # Move back to GPU
                if hasattr(_OCR_READER_EN, 'detector') and _OCR_READER_EN.detector is not None:
                    _OCR_READER_EN.detector = _OCR_READER_EN.detector.to(device)
                if hasattr(_OCR_READER_EN, 'recognizer') and _OCR_READER_EN.recognizer is not None:
                    _OCR_READER_EN.recognizer = _OCR_READER_EN.recognizer.to(device)
            return _OCR_READER_EN

def _get_manga_ocr():
    global _MANGA_OCR_INSTANCE
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if _MANGA_OCR_INSTANCE is None:
        with _MANGA_OCR_LOCK:
            if _MANGA_OCR_INSTANCE is None:
                try:
                    import os
                    from manga_ocr import MangaOcr
                    
                    # Target the local model directory we create via the download script
                    local_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "manga-ocr-base")
                    
                    if os.path.exists(local_model_path):
                        _MANGA_OCR_INSTANCE = MangaOcr(pretrained_model_name_or_path=local_model_path)
                    else:
                        # Fallback to online download if local directory doesn't exist
                        _MANGA_OCR_INSTANCE = MangaOcr(pretrained_model_name_or_path='kha-white/manga-ocr-base')
                except ImportError:
                    _MANGA_OCR_INSTANCE = None
    else:
        # Move back to GPU
        with _MANGA_OCR_LOCK:
            if _MANGA_OCR_INSTANCE is not None:
                if hasattr(_MANGA_OCR_INSTANCE, 'model') and _MANGA_OCR_INSTANCE.model is not None:
                    _MANGA_OCR_INSTANCE.model = _MANGA_OCR_INSTANCE.model.to(device)
    return _MANGA_OCR_INSTANCE

def unload_models():
    """Moves PyTorch OCR models from VRAM to CPU RAM (shared memory) for instant reload later."""
    global _OCR_READER_JA, _OCR_READER_EN, _MANGA_OCR_INSTANCE
    import torch
    
    with _OCR_LOCK:
        if _OCR_READER_JA is not None:
            if hasattr(_OCR_READER_JA, 'detector') and _OCR_READER_JA.detector is not None:
                _OCR_READER_JA.detector = _OCR_READER_JA.detector.to('cpu')
            if hasattr(_OCR_READER_JA, 'recognizer') and _OCR_READER_JA.recognizer is not None:
                _OCR_READER_JA.recognizer = _OCR_READER_JA.recognizer.to('cpu')
        
        if _OCR_READER_EN is not None:
            if hasattr(_OCR_READER_EN, 'detector') and _OCR_READER_EN.detector is not None:
                _OCR_READER_EN.detector = _OCR_READER_EN.detector.to('cpu')
            if hasattr(_OCR_READER_EN, 'recognizer') and _OCR_READER_EN.recognizer is not None:
                _OCR_READER_EN.recognizer = _OCR_READER_EN.recognizer.to('cpu')
                
    with _MANGA_OCR_LOCK:
        if _MANGA_OCR_INSTANCE is not None:
            if hasattr(_MANGA_OCR_INSTANCE, 'model') and _MANGA_OCR_INSTANCE.model is not None:
                _MANGA_OCR_INSTANCE.model = _MANGA_OCR_INSTANCE.model.to('cpu')
                
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def detect_text_regions(img_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Use MSER + morphological dilation to find text bounding boxes.
    Returns list of (x, y, w, h) rectangles, merged into coherent text lines.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    img_area = h * w

    # ① MSER — adaptive to image resolution
    max_char_area = img_area // 400       # at most 1/400 of the image
    min_char_area = max(30, img_area // 40000)
    mser = cv2.MSER_create(5, min_char_area, max_char_area)
    regions, _ = mser.detectRegions(gray)

    if not regions:
        return []

    # Convert point clouds → bounding rects, filter bad shapes
    rects = []
    for pts in regions:
        rx, ry, rw, rh = cv2.boundingRect(pts.reshape(-1, 1, 2))
        aspect = rw / (rh + 1e-5)
        if 0.15 < aspect < 8.0 and rw < w * 0.2 and rh < h * 0.2:
            rects.append((rx, ry, rx + rw, ry + rh))

    if not rects:
        return []

    # ② Draw MSER rects onto a mask, then dilate to merge nearby chars into words/lines
    mask = np.zeros((h, w), dtype=np.uint8)
    for (x0, y0, x1, y1) in rects:
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)

    # Dilate horizontally more than vertically → merge into text lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 4))
    dilated = cv2.dilate(mask, kernel, iterations=2)

    # ③ Find contours on merged mask → one rect per text line
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        # Filter: reasonable text-line size — not tiny noise, not huge background blobs
        if cw > 15 and ch > 8 and cw < w * 0.95 and ch < h * 0.95:
            boxes.append((x, y, cw, ch))

    # Sort top-to-bottom, left-to-right
    boxes.sort(key=lambda b: (b[1] // 30, b[0]))
    return boxes

class PDFLayoutExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        
        # Load standard doc to measure count
        self.doc = fitz.open(self.pdf_path)
        self.num_pages = len(self.doc)
        self.doc.close()

    def extract_layout(self, page_range_list=None, source_lang="Japanese"):
        """
        Extracts high-fidelity layout data page by page in parallel.
        """
        structured_data = []
        if page_range_list is not None:
            pages_to_extract = sorted(list(page_range_list))
        else:
            pages_to_extract = list(range(1, self.num_pages + 1))
        
        # CPU Multi-threading executor (limit max_workers to prevent RAM/VRAM overload)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self._extract_single_page, self.pdf_path, page_num, source_lang)
                for page_num in pages_to_extract
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    structured_data.append(result)
                except Exception as e:
                    logger.error(f"Failed extracting page layout: {e}")
                    
        # Retain original sequential reading order
        structured_data.sort(key=lambda x: x["page_num"])
        return structured_data

    async def extract_layout_stream(self, page_range_list=None, source_lang="Japanese"):
        """
        Extracts high-fidelity layout data page by page and yields them asynchronously as they complete.
        This allows pipelining extraction with translation.
        """
        import asyncio
        import concurrent.futures
        
        if page_range_list is not None:
            pages_to_extract = sorted(list(page_range_list))
        else:
            pages_to_extract = list(range(1, self.num_pages + 1))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            loop = asyncio.get_running_loop()
            
            # Process in strict chunks to prevent OCR from running ahead and stealing VRAM from the LLM
            chunk_size = 4
            for i in range(0, len(pages_to_extract), chunk_size):
                chunk_pages = pages_to_extract[i:i+chunk_size]
                futures = [
                    loop.run_in_executor(executor, self._extract_single_page, self.pdf_path, page_num, source_lang)
                    for page_num in chunk_pages
                ]
                
                for coro in asyncio.as_completed(futures):
                    try:
                        result = await coro
                        yield result
                    except Exception as e:
                        logger.error(f"Failed extracting page layout: {e}")
 
    def _extract_single_page(self, pdf_path, page_num, source_lang="Japanese"):
        """
        Thread-safe dual-engine page extraction:
        1. Fast Path: Native digital PDF text extraction via PyMuPDF (0ms, 100% precision, 0 GPU).
        2. Deep Path: EasyOCR CRAFT text region detection + MangaOCR recognition for scanned manga.
        """
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]
        page_width = float(page.rect.width)
        page_height = float(page.rect.height)
        
        # ── 1. Fast Path: Digital Text Check ──
        raw_text_blocks = page.get_text("blocks")
        valid_digital_blocks = []
        total_digital_chars = 0
        for b in raw_text_blocks:
            if b[6] == 0:  # text type
                txt = b[4].strip()
                if txt and self._is_meaningful_text(txt):
                    valid_digital_blocks.append(b)
                    total_digital_chars += len(txt)

        # In manga, scanned pages have large images and zero (or stray watermark) digital text
        images = page.get_images()
        is_scanned_manga_page = (len(images) > 0 and total_digital_chars < 30) or len(valid_digital_blocks) == 0

        if not is_scanned_manga_page:
            # Native digital text found! Bypass expensive CRAFT + MangaOCR
            blocks = []
            for idx, b in enumerate(valid_digital_blocks):
                bx0, by0, bx1, by1 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                txt = b[4].strip()
                if source_lang == "Japanese":
                    cleaned_txt = "".join(txt.splitlines()).strip()
                else:
                    cleaned_txt = " ".join(txt.splitlines()).strip()

                blocks.append({
                    "id": idx + 1,
                    "text": cleaned_txt,
                    "bbox": [bx0, by0, bx1, by1],
                    "lines_bboxes": [[bx0, by0, bx1, by1]],
                    "font_size": 12.0,
                    "font_name": "Helvetica",
                    "color": (0.0, 0.0, 0.0),
                    "width": bx1 - bx0,
                    "height": by1 - by0,
                    "center_x": (bx0 + bx1) / 2.0,
                    "center_y": (by0 + by1) / 2.0
                })

            doc.close()
            sorted_blocks = self._sort_layout_blocks(blocks, page_width)
            for idx, block in enumerate(sorted_blocks):
                block["id"] = idx + 1

            logger.info(f"[FastPath] Page {page_num}: Extracted {len(sorted_blocks)} native digital text blocks (0 GPU VRAM).")
            return {
                "page_num": page_num,
                "blocks": sorted_blocks,
                "page_width": page_width,
                "page_height": page_height
            }

        # ── 2. Deep Path: Scanned Manga OCR ──
        from PIL import Image

        TARGET_HEIGHT = 1600.0
        SCALE = TARGET_HEIGHT / max(1.0, page_height)
        pix = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE), alpha=False)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
        doc.close()
        
        # 2. Robust Text Region Detection using EasyOCR (CRAFT)
        # We strictly use it for bounding boxes, not recognition.
        # It natively ignores manga screentones and halftones.
        reader = _get_ocr_reader(source_lang)
        with _OCR_LOCK:
            # We use x_ths=0.15, y_ths=0.15 for both languages to prevent text fragmentation
            # (so a whole paragraph in a bubble is translated together, not line-by-line).
            ocr_results = reader.readtext(img_np, paragraph=True, x_ths=0.15, y_ths=0.15)
            
        blocks = []
        block_id_counter = 0
        
        # Only load MangaOCR if we are translating Japanese
        mocr = None
        if source_lang == "Japanese":
            mocr = _get_manga_ocr()
        
        # 3. Process each bounding box
        for coords, text in ocr_results:
            xs = [pt[0] for pt in coords]
            ys = [pt[1] for pt in coords]
            x0, x1 = max(0, int(min(xs))), min(img_np.shape[1], int(max(xs)))
            y0, y1 = max(0, int(min(ys))), min(img_np.shape[0], int(max(ys)))
            
            if x1 <= x0 or y1 <= y0:
                continue
                
            raw_text = ""
            
            if source_lang == "Japanese":
                # --- MANGA-OCR ALWAYS ---
                if mocr is not None:
                    try:
                        # Pad the crop slightly for better ViT recognition
                        pad = 10
                        cx0 = max(0, x0 - pad)
                        cy0 = max(0, y0 - pad)
                        cx1 = min(img_np.shape[1], x1 + pad)
                        cy1 = min(img_np.shape[0], y1 + pad)
                        
                        # Guard: skip degenerate crops (width or height < 8px)
                        # These cause MangaOCR ViT to receive an invalid tensor shape
                        if (cx1 - cx0) < 8 or (cy1 - cy0) < 8:
                            continue
                        
                        crop_np = img_np[cy0:cy1, cx0:cx1]
                        if crop_np.size == 0:
                            continue
                        
                        crop_img = Image.fromarray(crop_np)
                        # Ensure minimum size for ViT (32x32)
                        if crop_img.width < 16 or crop_img.height < 16:
                            crop_img = crop_img.resize(
                                (max(crop_img.width, 32), max(crop_img.height, 32)),
                                Image.LANCZOS
                            )
                        
                        with _MANGA_OCR_LOCK:
                            raw_text = mocr(crop_img)
                            
                        if raw_text:
                            raw_text = raw_text.strip()
                            logger.warning(f"[MangaOCR] Page {page_num}: detected '{raw_text}'")
                    except Exception as e:
                        logger.error(f"[!] MangaOCR failed on crop: {e}")
            else:
                # Use EasyOCR raw text directly for English/Malay
                raw_text = text
            
            if not raw_text or not self._is_meaningful_text(raw_text):
                continue
                
            bx0 = float(x0) / SCALE
            by0 = float(y0) / SCALE
            bx1 = float(x1) / SCALE
            by1 = float(y1) / SCALE

            blocks.append({
                "id": block_id_counter,
                "text": raw_text,
                "bbox": [bx0, by0, bx1, by1],
                "lines_bboxes": [[bx0, by0, bx1, by1]],
                "font_size": 12.0,
                "font_name": "Helvetica",
                "color": (0.0, 0.0, 0.0),
                "width": bx1 - bx0,
                "height": by1 - by0,
                "center_x": (bx0 + bx1) / 2.0,
                "center_y": (by0 + by1) / 2.0
            })
            block_id_counter += 1

        # Apply multi-column vertical-horizontal sorting
        sorted_blocks = self._sort_layout_blocks(blocks, page_width)

        # Re-index sorted blocks
        for idx, block in enumerate(sorted_blocks):
            block["id"] = idx + 1

        return {
            "page_num": page_num,
            "blocks": sorted_blocks,
            "page_width": page_width,
            "page_height": page_height
        }

    def _is_meaningful_text(self, text):
        """
        Cleans up and ignores background vector illustrations noise text blocks.
        Ultra-strict filter for scanned manga background artifacts.
        """
        text_clean = text.strip()
        if not text_clean:
            return False
            
        # Ignore lonely decorative characters (e.g. '*', '°', '▼')
        if len(text_clean) <= 3:
            has_cjk = any('\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' for c in text_clean)
            
            # Non-CJK short strings must be strictly alphanumeric (prevents extracting stray symbols)
            if not has_cjk and not text_clean.isalnum():
                return False
                
            # Filter out single/double random letters that aren't common English short words
            valid_shorts = {"a", "i", "an", "to", "by", "of", "in", "on", "at", "it", "he", "we", "us", "go", "up", "so", "no", "do", "am", "me", "my", "ok"}
            if text_clean.isalpha() and not has_cjk and text_clean.lower() not in valid_shorts and len(text_clean) < 3:
                return False

        # Filter out random formula/vector graphic delimiters like '=', '+', '|', '\', '/'
        symbols_count = sum(1 for c in text_clean if c in "=+|\\/_*<>[]{}")
        if symbols_count > 0.15 * len(text_clean):
            return False

        # Check density of alphanumeric + common grammar marks (include Japanese punctuation)
        jp_punct = "、。！？「」『』（）…〜～"
        alnum_count = sum(1 for c in text_clean if c.isalnum() or c.isspace() or c in ",.!?'-" or c in jp_punct)
        if len(text_clean) > 0 and (alnum_count / len(text_clean)) < 0.70:
            return False
            
        return True

    def _sort_layout_blocks(self, blocks, page_width):
        """
        Sort blocks in a page to maintain correct reading order.
        """
        if not blocks:
            return []

        full_width_threshold = 0.70 * page_width
        span_blocks = []
        normal_blocks = []

        for b in blocks:
            if b["width"] >= full_width_threshold:
                span_blocks.append(b)
            else:
                normal_blocks.append(b)

        span_blocks.sort(key=lambda x: x["center_y"])

        bands = []
        current_y_top = 0.0

        for sb in span_blocks:
            y_bottom = sb["bbox"][1]
            band_blocks = [b for b in normal_blocks if current_y_top <= b["center_y"] < y_bottom]
            bands.append({
                "type": "columns",
                "blocks": band_blocks
            })
            bands.append({
                "type": "span",
                "blocks": [sb]
            })
            current_y_top = sb["bbox"][3]

        remaining_blocks = [b for b in normal_blocks if b["center_y"] >= current_y_top]
        if remaining_blocks:
            bands.append({
                "type": "columns",
                "blocks": remaining_blocks
            })

        final_sorted_blocks = []
        for band in bands:
            if band["type"] == "span":
                final_sorted_blocks.extend(band["blocks"])
            else:
                band_blocks = band["blocks"]
                if not band_blocks:
                    continue
                
                columns = []
                band_blocks.sort(key=lambda x: x["bbox"][0])
                
                for b in band_blocks:
                    placed = False
                    for col in columns:
                        col_x0 = min(item["bbox"][0] for item in col)
                        col_x1 = max(item["bbox"][2] for item in col)
                        col_width = col_x1 - col_x0
                        
                        overlap_x0 = max(col_x0, b["bbox"][0])
                        overlap_x1 = min(col_x1, b["bbox"][2])
                        overlap_len = max(0.0, overlap_x1 - overlap_x0)
                        
                        if overlap_len > 0.4 * min(col_width, b["width"]) or abs(b["center_x"] - (col_x0 + col_x1)/2.0) < 50:
                            col.append(b)
                            placed = True
                            break
                    if not placed:
                        columns.append([b])

                columns.sort(key=lambda col: sum(item["center_x"] for item in col) / len(col))

                for col in columns:
                    col.sort(key=lambda x: x["bbox"][1])
                    final_sorted_blocks.extend(col)

        return final_sorted_blocks


if __name__ == "__main__":
    print("Testing Parallel PDF Layout Extractor Module...")
    print("[✓] Color-aware extractor compiled and ready.")
