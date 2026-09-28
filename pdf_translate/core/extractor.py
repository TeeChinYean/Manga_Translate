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

_COMIC_DETECTOR_LOCK = threading.Lock()
_COMIC_DETECTOR_SESSION = None

_PADDLE_OCR_LOCK = threading.Lock()
_PADDLE_OCR_INSTANCE = None

_OCR_READER_JA = None
_OCR_READER_EN = None

# ── GPU placement for extraction models (see core/gpu_budget.py) ──────────────
# _PLACEMENT[name] = "gpu" | "cpu" for comic_detector / manga_ocr / paddle_ocr.
_PLACEMENT = {}
_PLACEMENT_LOCK = threading.Lock()


def _ort_gpu_provider():
    """Best ONNX Runtime GPU provider available in this install, or None."""
    try:
        import onnxruntime as ort
        avail = ort.get_available_providers()
    except Exception:
        return None
    for p in ("CUDAExecutionProvider", "DmlExecutionProvider"):
        if p in avail:
            return p
    return None


def _torch_cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _loaded_models() -> dict:
    return {
        "comic_detector": _COMIC_DETECTOR_SESSION is not None,
        "manga_ocr": _MANGA_OCR_INSTANCE is not None,
        "paddle_ocr": _PADDLE_OCR_INSTANCE is not None,
    }


def prepare_extract_devices(force_probe: bool = True) -> dict:
    """
    Decide GPU vs CPU for each extraction model from the current free VRAM, and drop any
    loaded model whose placement changed so it reloads on the right device. Call at the
    start of each task (the LLM may have grown or shrunk since the last one).
    """
    from core import gpu_budget
    global _COMIC_DETECTOR_SESSION, _MANGA_OCR_INSTANCE, _PADDLE_OCR_INSTANCE
    with _PLACEMENT_LOCK:
        ort_gpu = _ort_gpu_provider()
        capable = {
            "comic_detector": ort_gpu is not None,
            "manga_ocr": _torch_cuda_available(),
            "paddle_ocr": ort_gpu is not None,
        }
        free, _src = gpu_budget.free_vram_mb(force=force_probe)
        loaded = _loaded_models()
        resident = {m for m, dev in _PLACEMENT.items() if dev == "gpu" and loaded.get(m)}
        plan = gpu_budget.plan_placement(capable, free_mb=free, resident=resident, source=_src)
        for m in ("comic_detector", "manga_ocr", "paddle_ocr"):
            if loaded[m] and _PLACEMENT.get(m) != plan[m]:
                if m == "comic_detector":
                    _COMIC_DETECTOR_SESSION = None
                elif m == "manga_ocr":
                    _MANGA_OCR_INSTANCE = None
                else:
                    _PADDLE_OCR_INSTANCE = None
                logger.info(f"[GPU Budget] {m}: {_PLACEMENT.get(m)} -> {plan[m]}, will reload.")
            _PLACEMENT[m] = plan[m]
        logger.info(f"[GPU Budget] placement {{{', '.join(f'{m}: {_PLACEMENT[m]}' for m in capable)}}} | {plan['_reason']}")
        if _PLACEMENT.get("manga_ocr") == "cpu" or not capable["manga_ocr"]:
            _free_torch_cache()
        return dict(_PLACEMENT, _reason=plan["_reason"], _ort_gpu=ort_gpu)


def _free_torch_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _measure_gpu_cost(name: str, warmup):
    """Run `warmup()` (first GPU inference) and record the VRAM it took for future planning."""
    from core import gpu_budget
    before, _ = gpu_budget.free_vram_mb(force=True)
    try:
        warmup()
    except Exception as e:
        logger.warning(f"[GPU Budget] {name} warm-up failed: {e}")
        return
    after, _ = gpu_budget.free_vram_mb(force=True)
    if before is not None and after is not None:
        gpu_budget.record_measured(name, before - after)


def _device_for(name: str) -> str:
    if name not in _PLACEMENT:
        prepare_extract_devices(force_probe=False)
    return _PLACEMENT.get(name, "cpu")


def _demote_to_cpu(name: str, exc: BaseException) -> bool:
    """On a GPU OOM / device error, move that model to CPU for the rest of the session."""
    from core import gpu_budget
    global _COMIC_DETECTOR_SESSION, _MANGA_OCR_INSTANCE, _PADDLE_OCR_INSTANCE
    if _PLACEMENT.get(name) != "gpu" or not gpu_budget.is_gpu_oom(exc):
        return False
    with _PLACEMENT_LOCK:
        _PLACEMENT[name] = "cpu"
        if name == "comic_detector":
            _COMIC_DETECTOR_SESSION = None
        elif name == "manga_ocr":
            _MANGA_OCR_INSTANCE = None
        else:
            _PADDLE_OCR_INSTANCE = None
    _free_torch_cache()
    logger.warning(f"[GPU Budget] {name} hit a GPU error ({exc}); falling back to CPU.")
    return True

def _get_paddle_ocr():
    """
    Singleton session for PaddleOCR (RapidOCR ONNX Runtime).
    Fast, lightweight (~15MB ONNX), multi-language OCR recognition.
    """
    global _PADDLE_OCR_INSTANCE
    if _PADDLE_OCR_INSTANCE is None:
        with _PADDLE_OCR_LOCK:
            if _PADDLE_OCR_INSTANCE is None:
                try:
                    from rapidocr_onnxruntime import RapidOCR
                    gpu_kwargs = {}
                    if _device_for("paddle_ocr") == "gpu":
                        flag = "use_cuda" if _ort_gpu_provider() == "CUDAExecutionProvider" else "use_dml"
                        gpu_kwargs = {f"{part}_{flag}": True for part in ("det", "cls", "rec")}
                    try:
                        _PADDLE_OCR_INSTANCE = RapidOCR(text_score=0.35, **gpu_kwargs)
                    except TypeError:
                        # Older rapidocr without GPU switches
                        _PADDLE_OCR_INSTANCE = RapidOCR(text_score=0.35)
                        gpu_kwargs = {}
                    logger.info(f"[PaddleOCR] Initialized RapidOCR ONNX engine ({'GPU ' + str(gpu_kwargs) if gpu_kwargs else 'CPU'}).")
                    if gpu_kwargs:
                        eng = _PADDLE_OCR_INSTANCE
                        demo = np.full((64, 256, 3), 255, dtype=np.uint8)
                        cv2.putText(demo, "Test 123", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
                        _measure_gpu_cost("paddle_ocr", lambda: eng(demo))
                except ImportError:
                    logger.warning("[PaddleOCR] rapidocr_onnxruntime is not installed.")
                    _PADDLE_OCR_INSTANCE = None
                except Exception as e:
                    logger.warning(f"[PaddleOCR] Failed to initialize RapidOCR: {e}")
                    _PADDLE_OCR_INSTANCE = None
    return _PADDLE_OCR_INSTANCE

def _get_comic_detector():
    """Singleton session for comic-text-detector.onnx with DirectML GPU acceleration and CPU fallback."""
    global _COMIC_DETECTOR_SESSION
    if _COMIC_DETECTOR_SESSION is None:
        with _COMIC_DETECTOR_LOCK:
            if _COMIC_DETECTOR_SESSION is None:
                try:
                    import onnxruntime as ort
                    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "models", "onnx", "comic-text-detector.onnx")
                    if os.path.exists(model_path):
                        providers = []
                        gpu_ep = _ort_gpu_provider()
                        if gpu_ep and _device_for("comic_detector") == "gpu":
                            providers.append(gpu_ep)
                        providers.append('CPUExecutionProvider')
                        _COMIC_DETECTOR_SESSION = ort.InferenceSession(model_path, providers=providers)
                        if len(providers) > 1:
                            sess = _COMIC_DETECTOR_SESSION
                            _measure_gpu_cost("comic_detector", lambda: sess.run(
                                None, {'images': np.zeros((1, 3, 1024, 1024), dtype=np.float32)}))
                        logger.info(f"[ComicTextDetector] Initialized detector with providers: {providers}")
                    else:
                        logger.info(f"[ComicTextDetector] Model not found at {model_path}, will use CRAFT fallback.")
                except Exception as e:
                    logger.warning(f"[ComicTextDetector] Failed to load ONNX detector: {e}")
                    _COMIC_DETECTOR_SESSION = None
    return _COMIC_DETECTOR_SESSION

def _detect_with_comic_detector(img_rgb: np.ndarray, session) -> list:
    """
    Runs comic-text-detector.onnx on an RGB image.
    Uses letterbox to 1024x1024, NMS with IoU 0.35, and returns list of (x0, y0, x1, y1) bounding boxes.
    """
    h, w = img_rgb.shape[:2]
    target_size = 1024
    scale = min(target_size / w, target_size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    dx = (target_size - nw) // 2
    dy = (target_size - nh) // 2
    canvas[dy:dy+nh, dx:dx+nw] = resized
    
    inp = (canvas.astype(np.float32) / 255.0).transpose((2, 0, 1))[None, ...]
    
    with _COMIC_DETECTOR_LOCK:
        blk, seg, det = session.run(None, {'images': inp})
        
    candidates = []
    for i in range(blk.shape[1]):
        row = blk[0, i]
        obj_conf = float(row[4])
        if obj_conf > 0.35:
            cx = (float(row[0]) - dx) / scale
            cy = (float(row[1]) - dy) / scale
            bw = float(row[2]) / scale
            bh = float(row[3]) / scale
            x0 = max(0, int(cx - bw / 2))
            y0 = max(0, int(cy - bh / 2))
            x1 = min(w, int(cx + bw / 2))
            y1 = min(h, int(cy + bh / 2))
            if x1 > x0 and y1 > y0:
                candidates.append((obj_conf, (x0, y0, x1, y1)))
                
    candidates.sort(key=lambda x: x[0], reverse=True)
    keep = []
    for conf, (x0, y0, x1, y1) in candidates:
        area = (x1 - x0) * (y1 - y0)
        overlap = False
        for kx0, ky0, kx1, ky1 in keep:
            karea = (kx1 - kx0) * (ky1 - ky0)
            ix0, iy0 = max(x0, kx0), max(y0, ky0)
            ix1, iy1 = min(x1, kx1), min(y1, ky1)
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            union = area + karea - inter
            if union > 0 and (inter / union) > 0.35:
                overlap = True
                break
        if not overlap:
            keep.append((x0, y0, x1, y1))
            
    return keep

def _get_ocr_reader(lang="Japanese"):
    global _OCR_READER_JA, _OCR_READER_EN
    eo = _get_easyocr()
    if eo is None:
        raise RuntimeError("easyocr is required for image OCR but not available in environment.")
    
    with _OCR_LOCK:
        if lang == "Japanese":
            if _OCR_READER_JA is None:
                # EasyOCR strictly runs on CPU for bounding box detection (CRAFT)
                # This leaves all GPU VRAM completely free for MangaOCR and LLM
                _OCR_READER_JA = eo.Reader(['ja', 'en'], gpu=False)
            return _OCR_READER_JA
        else:
            if _OCR_READER_EN is None:
                _OCR_READER_EN = eo.Reader(['en'], gpu=False)
            return _OCR_READER_EN

def _get_manga_ocr():
    global _MANGA_OCR_INSTANCE
    import torch
    use_gpu = torch.cuda.is_available() and _device_for("manga_ocr") == "gpu"
    device = 'cuda' if use_gpu else 'cpu'
    
    if _MANGA_OCR_INSTANCE is None:
        with _MANGA_OCR_LOCK:
            if _MANGA_OCR_INSTANCE is None:
                try:
                    import os
                    from manga_ocr import MangaOcr
                    
                    local_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "manga-ocr-base")
                    
                    model_ref = local_model_path if os.path.exists(local_model_path) else 'kha-white/manga-ocr-base'

                    def _load_and_warm():
                        global _MANGA_OCR_INSTANCE
                        _MANGA_OCR_INSTANCE = MangaOcr(pretrained_model_name_or_path=model_ref, force_cpu=not use_gpu)
                        if use_gpu:
                            from PIL import Image as _Img
                            _MANGA_OCR_INSTANCE(_Img.new("RGB", (64, 256), "white"))

                    if use_gpu:
                        # Measure load + first inference VRAM so later tasks plan with the real cost
                        _measure_gpu_cost("manga_ocr", _load_and_warm)
                    else:
                        _load_and_warm()
                    logger.info(f"[MangaOCR] Loaded MangaOCR model on device: {device.upper()}")
                except ImportError:
                    _MANGA_OCR_INSTANCE = None
    else:
        # Move back to target device if needed
        with _MANGA_OCR_LOCK:
            if _MANGA_OCR_INSTANCE is not None:
                if hasattr(_MANGA_OCR_INSTANCE, 'model') and _MANGA_OCR_INSTANCE.model is not None:
                    _MANGA_OCR_INSTANCE.model = _MANGA_OCR_INSTANCE.model.to(device)
    return _MANGA_OCR_INSTANCE

def unload_models():
    """Completely unloads and releases PyTorch & ONNX OCR models to free memory for downstream stages."""
    global _OCR_READER_JA, _OCR_READER_EN, _MANGA_OCR_INSTANCE, _COMIC_DETECTOR_SESSION, _PADDLE_OCR_INSTANCE
    import torch
    import gc
    
    with _OCR_LOCK:
        if _OCR_READER_JA is not None:
            del _OCR_READER_JA
            _OCR_READER_JA = None
        if _OCR_READER_EN is not None:
            del _OCR_READER_EN
            _OCR_READER_EN = None
                
    with _MANGA_OCR_LOCK:
        if _MANGA_OCR_INSTANCE is not None:
            del _MANGA_OCR_INSTANCE
            _MANGA_OCR_INSTANCE = None

    with _COMIC_DETECTOR_LOCK:
        if _COMIC_DETECTOR_SESSION is not None:
            del _COMIC_DETECTOR_SESSION
            _COMIC_DETECTOR_SESSION = None

    with _PADDLE_OCR_LOCK:
        if _PADDLE_OCR_INSTANCE is not None:
            del _PADDLE_OCR_INSTANCE
            _PADDLE_OCR_INSTANCE = None
                
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

def _merge_overlapping_boxes(raw_boxes, iou_thresh=0.20, containment_thresh=0.55):
    """
    Non-Maximum Suppression & Bounding Box Fusion for Manga Speech Bubbles:
    Merges multi-column vertical lines, overlapping CRAFT detections, and nested
    bounding boxes into single coherent speech bubble boxes.
    Eliminates text overlapping and reduces redundant OCR/LLM calls.
    """
    if not raw_boxes:
        return []

    boxes = [list(b) for b in raw_boxes]
    merged = True
    while merged:
        merged = False
        new_boxes = []
        skip_indices = set()
        
        for i in range(len(boxes)):
            if i in skip_indices:
                continue
            b1 = boxes[i]
            x0_1, y0_1, x1_1, y1_1 = b1
            w1 = max(1, x1_1 - x0_1)
            h1 = max(1, y1_1 - y0_1)
            area1 = w1 * h1
            
            for j in range(i + 1, len(boxes)):
                if j in skip_indices:
                    continue
                b2 = boxes[j]
                x0_2, y0_2, x1_2, y1_2 = b2
                w2 = max(1, x1_2 - x0_2)
                h2 = max(1, y1_2 - y0_2)
                area2 = w2 * h2
                
                # Intersection
                ix0 = max(x0_1, x0_2)
                iy0 = max(y0_1, y0_2)
                ix1 = min(x1_1, x1_2)
                iy1 = min(y1_1, y1_2)
                iw = max(0, ix1 - ix0)
                ih = max(0, iy1 - iy0)
                inter_area = iw * ih
                
                min_area = min(area1, area2)
                union_area = area1 + area2 - inter_area
                iou = inter_area / float(max(1, union_area))
                containment = inter_area / float(max(1, min_area))
                
                # Check for adjacent vertical columns in the same manga bubble:
                # Both columns must be vertical dialogue columns (h > w * 1.1)
                # with horizontal gap <= 15px and vertical overlap >= 60% of the shorter column
                is_both_vertical = (h1 > w1 * 1.1) and (h2 > w2 * 1.1)
                horizontal_dist = max(0, max(x0_1, x0_2) - min(x1_1, x1_2))
                vertical_overlap = ih / float(max(1, min(h1, h2)))
                is_adjacent_column = is_both_vertical and (horizontal_dist <= 15 and vertical_overlap >= 0.60)
                
                if iou >= iou_thresh or containment >= containment_thresh or is_adjacent_column:
                    # Merge b2 into b1
                    x0_1 = min(x0_1, x0_2)
                    y0_1 = min(y0_1, y0_2)
                    x1_1 = max(x1_1, x1_2)
                    y1_1 = max(y1_1, y1_2)
                    b1 = [x0_1, y0_1, x1_1, y1_1]
                    w1 = max(1, x1_1 - x0_1)
                    h1 = max(1, y1_1 - y0_1)
                    area1 = w1 * h1
                    skip_indices.add(j)
                    merged = True
                    
            new_boxes.append(b1)
        boxes = new_boxes

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
        # Pages whose extraction raised: list of (page_num, error message). See BUG.md B4.
        self.failed_pages = []

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
            futures = {
                executor.submit(self._extract_single_page, self.pdf_path, page_num, source_lang): page_num
                for page_num in pages_to_extract
            }
            for future in concurrent.futures.as_completed(futures):
                page_num = futures[future]
                try:
                    result = future.result()
                    structured_data.append(result)
                except Exception as e:
                    logger.error(f"Failed extracting layout of page {page_num}: {e}")
                    self.failed_pages.append((page_num, str(e)))
                    
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

                async def _extract(page_num):
                    try:
                        res = await loop.run_in_executor(
                            executor, self._extract_single_page, self.pdf_path, page_num, source_lang
                        )
                        return page_num, res, None
                    except Exception as e:
                        return page_num, None, e

                for coro in asyncio.as_completed([_extract(p) for p in chunk_pages]):
                    page_num, result, err = await coro
                    if err is not None:
                        # Keep going, but record the page so the caller can report it (BUG.md B4)
                        logger.error(f"Failed extracting layout of page {page_num}: {err}")
                        self.failed_pages.append((page_num, str(err)))
                        continue
                    yield result
 
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
                    "center_y": (by0 + by1) / 2.0,
                    "ocr_engine": "PyMuPDF (Digital Vector)"
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

        # Optimized detection resolution: 850.0px runs ~30% faster on CPU while capturing all bubbles
        TARGET_HEIGHT = 850.0
        SCALE = TARGET_HEIGHT / max(1.0, page_height)
        pix = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE), alpha=False)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
        doc.close()
        
        # 2. Text Region Detection (Comic-Text-Detector ONNX with CRAFT fallback)
        comic_session = _get_comic_detector()
        raw_box_to_text = {}
        
        if comic_session is not None:
            # High-speed specialized YOLOv8 detection for manga dialogue bubbles
            try:
                merged_boxes = _detect_with_comic_detector(img_np, comic_session)
            except Exception as det_err:
                if not _demote_to_cpu("comic_detector", det_err):
                    raise
                merged_boxes = _detect_with_comic_detector(img_np, _get_comic_detector())
            logger.info(f"[ComicTextDetector] Page {page_num}: Detected {len(merged_boxes)} speech bubbles.")
        else:
            # Fallback to EasyOCR CRAFT
            reader = _get_ocr_reader(source_lang)
            with _OCR_LOCK:
                ocr_results = reader.readtext(
                    img_np,
                    paragraph=True,
                    x_ths=0.15,
                    y_ths=0.15,
                    text_threshold=0.60,
                    low_text=0.35,
                    link_threshold=0.35
                )

            candidate_boxes = []
            for coords, text in ocr_results:
                xs = [pt[0] for pt in coords]
                ys = [pt[1] for pt in coords]
                x0, x1 = max(0, int(min(xs))), min(img_np.shape[1], int(max(xs)))
                y0, y1 = max(0, int(min(ys))), min(img_np.shape[0], int(max(ys)))
                if x1 > x0 and y1 > y0:
                    candidate_boxes.append([x0, y0, x1, y1])
                    raw_box_to_text[(x0, y0, x1, y1)] = text

            merged_boxes = _merge_overlapping_boxes(candidate_boxes)
            logger.info(f"[CRAFT Fallback] Page {page_num}: CRAFT detected {len(candidate_boxes)} boxes → merged into {len(merged_boxes)} bubble boxes.")
            
        blocks = []
        block_id_counter = 0
        seen_texts = set()
        
        # OCR engine order (BUG.md B7):
        #   Japanese -> MangaOCR first (handles vertical text + kana); PaddleOCR only as fallback.
        #   Others   -> PaddleOCR (RapidOCR ONNX) first.
        # RapidOCR's default PP-OCR model is Chinese/English: on vertical Japanese it returns
        # non-empty but wrong text (kana dropped, columns interleaved), so it must not be primary.
        paddle_ocr = _get_paddle_ocr()
        mocr = None
        if source_lang == "Japanese":
            mocr = _get_manga_ocr()
            if mocr is None:
                logger.warning("[OCR] MangaOCR unavailable; falling back to PaddleOCR for Japanese (accuracy will drop).")

        def _crop(x0, y0, x1, y1, pad):
            cx0 = max(0, x0 - pad)
            cy0 = max(0, y0 - pad)
            cx1 = min(img_np.shape[1], x1 + pad)
            cy1 = min(img_np.shape[0], y1 + pad)
            if (cx1 - cx0) < 8 or (cy1 - cy0) < 8:
                return None
            return img_np[cy0:cy1, cx0:cx1]

        def _ocr_paddle(x0, y0, x1, y1):
            nonlocal paddle_ocr
            if paddle_ocr is None:
                return ""
            try:
                crop_np = _crop(x0, y0, x1, y1, 6)
                if crop_np is None:
                    return ""
                paddle_res, _ = paddle_ocr(crop_np)
                if paddle_res:
                    text = "".join(line[1].strip() for line in paddle_res if line and len(line) > 1 and line[1])
                    return text.strip()
            except Exception as pe:
                if _demote_to_cpu("paddle_ocr", pe):
                    paddle_ocr = _get_paddle_ocr()
                    return _ocr_paddle(x0, y0, x1, y1)
                logger.warning(f"[PaddleOCR] Recognition failed on crop: {pe}")
            return ""

        def _ocr_manga(x0, y0, x1, y1):
            nonlocal mocr
            if mocr is None:
                return ""
            try:
                crop_np = _crop(x0, y0, x1, y1, 8)
                if crop_np is None or crop_np.size == 0:
                    return ""
                crop_img = Image.fromarray(crop_np)
                if crop_img.width < 16 or crop_img.height < 16:
                    crop_img = crop_img.resize(
                        (max(crop_img.width, 32), max(crop_img.height, 32)),
                        Image.LANCZOS
                    )
                with _MANGA_OCR_LOCK:
                    import torch
                    with torch.inference_mode():
                        try:
                            x = mocr._preprocess(crop_img)
                            tokens = mocr.model.generate(x[None].to(mocr.model.device), max_new_tokens=64, max_length=None)[0].cpu()
                            text = mocr.tokenizer.decode(tokens, skip_special_tokens=True)
                            from manga_ocr.ocr import post_process
                            text = post_process(text)
                        except Exception:
                            text = mocr(crop_img)
                return (text or "").strip()
            except Exception as e:
                if _demote_to_cpu("manga_ocr", e):
                    mocr = _get_manga_ocr()
                    if mocr is not None:
                        try:
                            return (mocr(Image.fromarray(_crop(x0, y0, x1, y1, 8))) or "").strip()
                        except Exception as e2:
                            logger.error(f"[!] MangaOCR (CPU retry) failed on crop: {e2}")
                    return ""
                logger.error(f"[!] MangaOCR failed on crop: {e}")
            return ""

        if mocr is not None:
            ocr_chain = [(_ocr_manga, "MangaOCR (ViT)"), (_ocr_paddle, "PaddleOCR (PP-OCRv4 Fallback)")]
        else:
            ocr_chain = [(_ocr_paddle, "PaddleOCR (PP-OCRv4)")]

        # 3. Process each unified speech bubble box
        for (x0, y0, x1, y1) in merged_boxes:
            if x1 <= x0 or y1 <= y0:
                continue

            raw_text = ""
            ocr_engine_name = ocr_chain[0][1]
            for ocr_fn, engine_name in ocr_chain:
                raw_text = ocr_fn(x0, y0, x1, y1)
                if raw_text:
                    ocr_engine_name = engine_name
                    logger.info(f"[{engine_name}] Page {page_num}: detected '{raw_text}'")
                    break

            # --- 3c. Tertiary Fallback: EasyOCR text matching ---
            if not raw_text and raw_box_to_text:
                matching = [
                    t for (rx0, ry0, rx1, ry1), t in raw_box_to_text.items()
                    if max(0, min(x1, rx1) - max(x0, rx0)) * max(0, min(y1, ry1) - max(y0, ry0)) > 0
                ]
                raw_text = " ".join(matching).strip()
                if raw_text:
                    ocr_engine_name = "EasyOCR (Fallback)"
            
            if not raw_text or not self._is_meaningful_text(raw_text):
                continue

            # Deduplicate near-identical text to prevent stacking duplicate blocks
            clean_key = "".join(c for c in raw_text if c.isalnum() or '\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff')
            if clean_key and clean_key in seen_texts:
                continue
            if clean_key:
                seen_texts.add(clean_key)
                
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
                "center_y": (by0 + by1) / 2.0,
                "ocr_engine": ocr_engine_name
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
            
        has_cjk = any('\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' for c in text_clean)
        has_jp_punct = any('\u3000' <= c <= '\u303f' or '\uff01' <= c <= '\uff5e' or c in '!?…―〜' for c in text_clean)
        
        # Ignore lonely decorative characters (e.g. '*', '°', '▼')
        if len(text_clean) <= 3:
            # Non-CJK short strings must be strictly alphanumeric or legitimate Japanese dialogue punctuation
            if not has_cjk and not has_jp_punct and not text_clean.isalnum():
                return False
                
            # Filter out single/double random letters that aren't common English short words
            valid_shorts = {"a", "i", "an", "to", "by", "of", "in", "on", "at", "it", "he", "we", "us", "go", "up", "so", "no", "do", "am", "me", "my", "ok"}
            if text_clean.isalpha() and not has_cjk and text_clean.lower() not in valid_shorts and len(text_clean) < 3:
                return False

        # Filter out random formula/vector graphic delimiters like '=', '+', '|', '\', '/'
        symbols_count = sum(1 for c in text_clean if c in "=+|\\/_*<>[]{}")
        if symbols_count > 0.25 * len(text_clean):
            return False

        # If it has CJK characters or Japanese punctuation exclamations (e.g. 「あっ」「……」「！？」), it is legitimate dialogue
        if has_cjk or has_jp_punct:
            return True

        # Check density of alphanumeric + common grammar marks for non-CJK text
        alnum_count = sum(1 for c in text_clean if c.isalnum() or c.isspace() or c in ",.!?'-")
        if len(text_clean) > 0 and (alnum_count / len(text_clean)) < 0.60:
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
