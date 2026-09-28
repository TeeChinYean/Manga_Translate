#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ ANTIGRAVITY - LaMa Inpainting PDF Renderer (V3 - Native Async & Correct API)
Pipeline per page:
  1. Render PDF page to high-res image (via PyMuPDF)
  2. Build text mask from block bboxes
  3. Run LaMa neural inpainting natively with correct InpainterConfig parameters
  4. Overlay translated CJK text with adaptive font size
  5. Embed the rendered image back as a PDF page
"""

import os
import sys
import asyncio
import threading
import time
import cv2
import numpy as np
import logging
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

_LAMA_SESSION_LOCK = threading.Lock()
_LAMA_SESSION = None

# ── LaMa CPU parallelism ──────────────────────────────────────────────────────
# ORT InferenceSession.run is thread-safe. Running 2 crops at once with half the threads
# each usually beats 1 crop with all threads on 8+ logical cores (better core utilisation).
_CPU = os.cpu_count() or 4
LAMA_PARALLEL = max(1, int(os.getenv("LAMA_PARALLEL", "2" if _CPU >= 8 else "1")))
LAMA_THREADS = max(1, int(os.getenv("LAMA_THREADS", str(max(2, (_CPU // 2) // LAMA_PARALLEL)))))
_LAMA_RUN_SEM = threading.BoundedSemaphore(LAMA_PARALLEL)

# ── Render timing breakdown (where does rendering time go?) ───────────────────
_RENDER_STATS = {}
_RENDER_STATS_LOCK = threading.Lock()


def _rstat(tag: str, seconds: float, n: int = 1):
    with _RENDER_STATS_LOCK:
        e = _RENDER_STATS.setdefault(tag, {"calls": 0, "seconds": 0.0})
        e["calls"] += n
        e["seconds"] = round(e["seconds"] + seconds, 2)


def reset_render_stats():
    with _RENDER_STATS_LOCK:
        _RENDER_STATS.clear()


def get_render_stats() -> dict:
    with _RENDER_STATS_LOCK:
        out = {k: dict(v) for k, v in _RENDER_STATS.items()}
    out["_config"] = {"cpu": _CPU, "lama_parallel": LAMA_PARALLEL, "lama_threads": LAMA_THREADS}
    return out

def _get_lama_session():
    """Singleton session for lama.onnx with CPU execution and ORT_DISABLE_ALL for FFC stability."""
    global _LAMA_SESSION
    if _LAMA_SESSION is None:
        with _LAMA_SESSION_LOCK:
            if _LAMA_SESSION is None:
                try:
                    import onnxruntime as ort
                    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "models", "onnx", "lama.onnx")
                    if os.path.exists(model_path):
                        opts = ort.SessionOptions()
                        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                        opts.intra_op_num_threads = LAMA_THREADS
                        # LaMa runs on CPU to guarantee FFC DFT node numerical stability
                        _LAMA_SESSION = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
                        logger.info(f"[LaMa] Loaded LaMa ONNX session on CPU with ORT_DISABLE_ALL")
                    else:
                        logger.info(f"[LaMa] Model not found at {model_path}, using Telea diffusion fallback.")
                except Exception as e:
                    logger.warning(f"[LaMa] Failed to load LaMa session: {e}")
                    _LAMA_SESSION = None
    return _LAMA_SESSION

# ── Font config ────────────────────────────────────────────────────────────────
_FONT_PATHS = [
    # Linux CJK fonts (Bold/Medium first for premium manga look)
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    # Windows CJK fonts
    "C:\\Windows\\Fonts\\msyhbd.ttc",
    "C:\\Windows\\Fonts\\msyh.ttc",
    "C:\\Windows\\Fonts\\simhei.ttf",
    # macOS CJK fonts
    "/System/Library/Fonts/PingFang.ttc",
    "/Library/Fonts/Arial Unicode.ttf"
]
_FONT_PATH = next((p for p in _FONT_PATHS if os.path.exists(p)), None)

# FreeType font objects are cached and shared; serialize text drawing across threads.
_DRAW_LOCK = threading.Lock()


def _save_jpeg(img: Image.Image, path: str):
    img.save(path, format="JPEG", quality=80, optimize=True)

def unload_models():
    """Cleans up memory/cache if needed."""
    global _LAMA_SESSION
    with _LAMA_SESSION_LOCK:
        if _LAMA_SESSION is not None:
            del _LAMA_SESSION
            _LAMA_SESSION = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ── Helpers ────────────────────────────────────────────────────────────────────
def _sample_bg(img: Image.Image, x0, y0, x1, y1) -> tuple:
    W, H = img.size
    pts = [
        (x0 - 4, y0 - 4), (x1 + 4, y0 - 4),
        (x0 - 4, y1 + 4), (x1 + 4, y1 + 4),
        ((x0 + x1) // 2, y0 - 4), ((x0 + x1) // 2, y1 + 4),
        (x0 - 4, (y0 + y1) // 2), (x1 + 4, (y0 + y1) // 2),
    ]
    colors = []
    for px, py in pts:
        cx = max(0, min(W - 1, int(px)))
        cy = max(0, min(H - 1, int(py)))
        colors.append(img.getpixel((cx, cy)))
    r = int(sum(c[0] for c in colors) / len(colors))
    g = int(sum(c[1] for c in colors) / len(colors))
    b = int(sum(c[2] for c in colors) / len(colors))
    return (r, g, b)


def _contrast_color(bg: tuple) -> str:
    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    return "black" if lum > 127 else "white"


def _wrap_cjk(text: str, font, max_w: int) -> list:
    CLOSING_PUNCT = {')', ']', '}', '>', '）', '】', '〉', '｝', '，', '。', '、', '；', '：', '？', '！', '”', '’'}
    lines, curr = [], ""
    for ch in text:
        test = curr + ch
        try:
            tw = font.getbbox(test)[2] - font.getbbox(test)[0]
        except Exception:
            tw = len(test) * 14
        if tw > max_w and curr:
            # If the current character is a closing punctuation, keep it on the current line to avoid orphan punctuation
            if ch in CLOSING_PUNCT:
                curr = test
            else:
                lines.append(curr)
                curr = ch
        else:
            curr = test
    if curr:
        lines.append(curr)
    return lines or [text]


from functools import lru_cache

@lru_cache(maxsize=128)
def _get_font(fs: int):
    try:
        return ImageFont.truetype(_FONT_PATH, fs) if _FONT_PATH else ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()

def _best_font(text: str, box_w: int, box_h: int, font_scale: float = 1.0):
    # Calculate usable area with 15% padding to prevent text from touching the borders
    pad_w = int(box_w * 0.15)
    pad_h = int(box_h * 0.15)
    usable_w = max(box_w - pad_w, 20)
    usable_h = max(box_h - pad_h, 20)
    
    text_len = max(len(text), 1)
    
    # Estimate max font size based on area (fs * fs * 1.25 * len <= w * h)
    import math
    estimated_max_fs = int(math.sqrt((usable_w * usable_h) / (text_len * 1.25)))
    max_fs = min(int(estimated_max_fs * 1.5), usable_w, usable_h, 120)
    min_fs = 10
    
    if max_fs <= min_fs:
        font = _get_font(min_fs)
        return font, _wrap_cjk(text, font, usable_w), min_fs
        
    low = min_fs
    high = max_fs
    best_font = _get_font(min_fs)
    best_lines = _wrap_cjk(text, best_font, usable_w)
    best_fs = min_fs
    
    # Binary search: O(log N) iterations (~7 steps) instead of O(N) linear decrements (~150 steps)
    while low <= high:
        mid = (low + high) // 2
        font = _get_font(mid)
        lines = _wrap_cjk(text, font, usable_w)
        line_height = mid * 1.25
        total_h = len(lines) * line_height
        
        fits = False
        if total_h <= usable_h:
            max_line_w = 0
            for line in lines:
                try:
                    lw = font.getbbox(line)[2] - font.getbbox(line)[0]
                except Exception:
                    lw = len(line) * mid
                if lw > max_line_w:
                    max_line_w = lw
            if max_line_w <= usable_w + 5:
                fits = True
                
        if fits:
            best_font = font
            best_lines = lines
            best_fs = mid
            low = mid + 1  # try a larger font
        else:
            high = mid - 1 # shrink font
            
    if abs(font_scale - 1.0) > 0.01:
        scaled_fs = max(8, min(120, int(best_fs * font_scale)))
        scaled_font = _get_font(scaled_fs)
        scaled_lines = _wrap_cjk(text, scaled_font, usable_w)
        return scaled_font, scaled_lines, scaled_fs

    return best_font, best_lines, best_fs


def _fast_telea_inpaint(img_pil: Image.Image, mask_pil: Image.Image) -> Image.Image:
    """
    Lightning-fast, texture-preserving inpainting using OpenCV Telea diffusion.
    - Zero heavy neural dependencies
    - <15ms execution time
    - Flawlessly diffuses screentones, line art, shading, and background textures
    """
    img_np = np.array(img_pil.convert("RGB"))
    mask_np = np.array(mask_pil.convert("L"))
    
    # Threshold mask to strictly 0 or 255
    _, mask_bin = cv2.threshold(mask_np, 10, 255, cv2.THRESH_BINARY)
    if np.count_nonzero(mask_bin) == 0:
        return img_pil

    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    inpainted_bgr = cv2.inpaint(img_bgr, mask_bin, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
    inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(inpainted_rgb)


LAMA_SIZE = 512          # fixed input size of lama.onnx
LAMA_CONTEXT_PAD = 48   # surrounding context (px) given to LaMa around each masked region
LAMA_MERGE_PX = 24      # strokes closer than this are treated as one region
LAMA_MAX_GROUP = 768    # max side of a merged crop; >512 is downscaled uniformly (<=1.5x). Fewer LaMa calls.


def _lama_regions(mask_bin: np.ndarray, img_w: int, img_h: int) -> list:
    """
    Group masked pixels into crop boxes (x0, y0, x1, y1) for LaMa.
    Nearby regions are greedily merged while the merged box still fits in
    LAMA_MAX_GROUP (<=1.5x uniform downscale); larger regions stay alone and
    are downscaled uniformly (aspect ratio preserved). Each LaMa call costs
    ~5-10s on CPU, so fewer calls matter (BUG.md B12).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (LAMA_MERGE_PX, LAMA_MERGE_PX))
    merged = cv2.dilate((mask_bin > 0).astype(np.uint8), kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)

    boxes = []
    for i in range(1, n):
        x, y, w, h = int(stats[i, 0]), int(stats[i, 1]), int(stats[i, 2]), int(stats[i, 3])
        boxes.append((
            max(0, x - LAMA_CONTEXT_PAD), max(0, y - LAMA_CONTEXT_PAD),
            min(img_w, x + w + LAMA_CONTEXT_PAD), min(img_h, y + h + LAMA_CONTEXT_PAD),
        ))
    boxes.sort(key=lambda b: (b[1], b[0]))

    groups = []
    for b in boxes:
        for gi, g in enumerate(groups):
            u = (min(g[0], b[0]), min(g[1], b[1]), max(g[2], b[2]), max(g[3], b[3]))
            if max(u[2] - u[0], u[3] - u[1]) <= LAMA_MAX_GROUP:
                groups[gi] = u
                break
        else:
            groups.append(b)
    return groups


def _expand_to_window(x0: int, y0: int, x1: int, y1: int, img_w: int, img_h: int) -> tuple:
    """
    Grow a box smaller than LAMA_SIZE into a LAMA_SIZE window centered on it
    (clamped to the image), so LaMa runs at native resolution with more context.
    """
    def grow(a0, a1, limit):
        size = min(LAMA_SIZE, limit)
        if a1 - a0 >= size:
            return a0, a1
        c = (a0 + a1) // 2
        n0 = max(0, min(c - size // 2, limit - size))
        return n0, n0 + size
    x0, x1 = grow(x0, x1, img_w)
    y0, y1 = grow(y0, y1, img_h)
    return x0, y0, x1, y1


def _lama_inpaint_crop(session, crop_rgb: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
    """Run LaMa on one crop: pad to square (reflect), uniform resize to LAMA_SIZE, then undo."""
    h, w = crop_rgb.shape[:2]
    side = max(h, w)
    pad_b, pad_r = side - h, side - w
    sq_img = cv2.copyMakeBorder(crop_rgb, 0, pad_b, 0, pad_r, cv2.BORDER_REFLECT_101) if (pad_b or pad_r) else crop_rgb
    sq_mask = cv2.copyMakeBorder(crop_mask, 0, pad_b, 0, pad_r, cv2.BORDER_CONSTANT, value=0) if (pad_b or pad_r) else crop_mask

    if side != LAMA_SIZE:
        interp = cv2.INTER_AREA if side > LAMA_SIZE else cv2.INTER_CUBIC
        img_in = cv2.resize(sq_img, (LAMA_SIZE, LAMA_SIZE), interpolation=interp)
        mask_in = cv2.resize(sq_mask, (LAMA_SIZE, LAMA_SIZE), interpolation=cv2.INTER_NEAREST)
    else:
        img_in, mask_in = sq_img, sq_mask
    mask_in = cv2.dilate(mask_in, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)

    lama_img_in = (img_in.astype(np.float32) / 255.0).transpose((2, 0, 1))[None, ...]
    lama_mask_in = (mask_in > 0).astype(np.float32)[None, None, ...]
    t_wait = time.time()
    with _LAMA_RUN_SEM:
        t_run = time.time()
        lama_out = session.run(None, {'l_image_': lama_img_in, 'l_mask_': lama_mask_in})[0]
        _rstat("lama_run", time.time() - t_run)
    _rstat("lama_wait", t_run - t_wait, n=0)
    out = np.clip(lama_out[0].transpose((1, 2, 0)), 0, 255).astype(np.uint8)

    if side != LAMA_SIZE:
        interp = cv2.INTER_CUBIC if side > LAMA_SIZE else cv2.INTER_AREA
        out = cv2.resize(out, (side, side), interpolation=interp)
    return out[:h, :w]


def _lama_inpaint(img_pil: Image.Image, mask_pil: Image.Image) -> Image.Image:
    """
    High-fidelity neural inpainting using LaMa ONNX (Fast Fourier Convolutions).
    Runs per masked region crop (not the whole page), so small bubbles are
    processed at native resolution without aspect distortion (BUG.md B2).
    Falls back gracefully to OpenCV Telea if LaMa is unavailable.
    """
    session = _get_lama_session()
    if session is None:
        return _fast_telea_inpaint(img_pil, mask_pil)

    try:
        img_np = np.array(img_pil.convert("RGB"))
        mask_np = np.array(mask_pil.convert("L"))

        _, mask_bin = cv2.threshold(mask_np, 10, 255, cv2.THRESH_BINARY)
        if np.count_nonzero(mask_bin) == 0:
            return img_pil

        h, w = img_np.shape[:2]
        result = img_np.copy()
        for (x0, y0, x1, y1) in _lama_regions(mask_bin, w, h):
            x0, y0, x1, y1 = _expand_to_window(x0, y0, x1, y1, w, h)
            crop_mask = mask_bin[y0:y1, x0:x1]
            if not np.any(crop_mask):
                continue
            healed = _lama_inpaint_crop(session, result[y0:y1, x0:x1], crop_mask)
            # Only replace masked pixels; everything else stays bit-exact original.
            sel = crop_mask > 0
            result[y0:y1, x0:x1][sel] = healed[sel]
        return Image.fromarray(result)
    except Exception as e:
        logger.warning(f"[LaMa] Inference error, falling back to Telea: {e}")
        return _fast_telea_inpaint(img_pil, mask_pil)


RING_PAD = 8             # px ring outside the text box used to judge the background
RING_WHITE_MIN = 0.92    # clean bubble: >=92% of ring pixels are white ...
RING_MID_MAX = 0.08      # ... and <=8% are mid-tone (screentone / shading)


def _ring_is_textured(gray: np.ndarray, x0: int, y0: int, x1: int, y1: int, pad: int = RING_PAD) -> bool:
    """
    Decide Telea vs LaMa from a thin ring *outside* the text box (BUG.md B11/B12).
    Text boxes are tight, so the box border itself crosses glyph strokes and looks
    "textured" even inside a clean white bubble; the outer ring does not.
    """
    H, W = gray.shape[:2]
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    region = gray[Y0:Y1, X0:X1]
    ring = np.ones(region.shape, dtype=bool)
    ring[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
    vals = region[ring]
    if vals.size < 16:
        return True
    white = float((vals >= 200).mean())
    mid = float(((vals > 60) & (vals < 200)).mean())
    return white < RING_WHITE_MIN or mid > RING_MID_MAX


def _create_stroke_mask(crop_rgb: np.ndarray, ink_thresh: int = 95, dilate_iter: int = 2, max_stroke_ratio: float = 0.35,
                        textured: bool = None) -> tuple:
    """
    Extracts precise text stroke contours (glyph pixels) instead of a blunt rectangle.
    Suppresses manga halftone/screentone noise and completely eliminates ghost character edges.
    Parameters allow real-time tuning from the sidebar:
      - ink_thresh: strictly filters glyph ink (default 95, range 50-160)
      - dilate_iter: stroke dilation radius (default 2, range 1-4)
      - max_stroke_ratio: protection ratio to avoid removing artwork/hair (default 0.35, range 0.15-0.50)
    Returns: (mask_np, bg_val)
    """
    h, w = crop_rgb.shape[:2]
    if h < 4 or w < 4:
        return np.zeros((h, w), dtype=np.uint8), 255.0, False
        
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    
    # 1. Edge/Border background estimation using median and variance (robust against bubble line intersection)
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    bg_val = float(np.median(border))
    bg_std = float(np.std(border))
    # True if the border region is dark or has screentone/halftone/texture variance
    is_textured = (bg_val < 170) or (bg_std > 22.0)
    if textured is not None:
        # Caller measured the surroundings of the box (see _ring_is_textured); trust it.
        is_textured = bool(textured)
    
    # 2. Suppress halftone screentone dots and mosquito noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    if bg_val >= 160:
        # Light background (speech bubble or bright scene)
        # Real text ink is distinctly darker than background; clamp to ink_thresh to prevent grabbing skin/clothing/lace
        thresh_val = min(bg_val - 45, float(ink_thresh))
        mask = (blurred < thresh_val).astype(np.uint8) * 255
    elif bg_val <= 90:
        # Dark background (night scene, black bubble, dark panel)
        # Real text ink is distinctly lighter than background
        thresh_val = max(bg_val + 45, float(255 - ink_thresh))
        mask = (blurred > thresh_val).astype(np.uint8) * 255
    else:
        # Midtone background (shading, textured drawing)
        # Use Otsu on the contrast to cleanly separate glyphs from screentones
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = otsu

    # 3. Filter out isolated tiny dots (halftone speckles < 4px)
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, clean_kernel)
    
    # 4. Dilate text strokes with dilate_iter iterations (3x3 ellipse) to safely cover anti-aliased character edges
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(cleaned_mask, dilate_kernel, iterations=max(1, int(dilate_iter)))
    
    stroke_ratio = np.count_nonzero(dilated) / float(h * w)
    
    # Protection against grabbing artwork/lace/hair:
    # Text in bubbles rarely exceeds max_stroke_ratio of box area.
    # If stroke_ratio > max_stroke_ratio, strictly clamp to deep black ink (< 60)
    if stroke_ratio > float(max_stroke_ratio):
        strict_mask = (blurred < 60).astype(np.uint8) * 255
        strict_cleaned = cv2.morphologyEx(strict_mask, cv2.MORPH_OPEN, clean_kernel)
        dilated = cv2.dilate(strict_cleaned, dilate_kernel, iterations=1)
        stroke_ratio = np.count_nonzero(dilated) / float(h * w)

    final_mask = dilated if (0.005 <= stroke_ratio <= 0.85) else mask
    if is_textured:
        final_mask = _add_outline_halo(final_mask, blurred, glyph_is_dark=(bg_val > 90))
    return final_mask, bg_val, is_textured


HALO_LIGHT_MIN = 180   # halo pixels around dark glyphs must be at least this bright
HALO_DARK_MAX = 75     # halo pixels around light glyphs must be at most this dark


def _add_outline_halo(glyph_mask: np.ndarray, gray: np.ndarray, glyph_is_dark: bool) -> np.ndarray:
    """
    Outlined lettering over artwork (e.g. black glyphs with a thick white rim, or
    white glyphs with a black rim) leaves the rim behind when only the glyph core is
    masked; the inpainter then redraws the glyph from the rim silhouette (BUG.md B11).
    Add the opposite-polarity rim that directly surrounds the glyph strokes.
    """
    h, w = glyph_mask.shape[:2]
    if not np.any(glyph_mask):
        return glyph_mask
    radius = int(np.clip(round(0.06 * min(h, w)), 3, 10))
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    near_glyph = cv2.dilate(glyph_mask, ring_kernel) > 0
    rim_color = (gray >= HALO_LIGHT_MIN) if glyph_is_dark else (gray <= HALO_DARK_MAX)
    halo = (near_glyph & rim_color).astype(np.uint8) * 255
    combined = cv2.bitwise_or(glyph_mask, halo)
    return cv2.dilate(combined, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)


# ── Main Renderer Class ────────────────────────────────────────────────────────
class PDFLayoutRenderer:
    def __init__(self, original_pdf_path=None, output_pdf_path=None,
                 ink_thresh: int = 95, dilate_iter: int = 2,
                 max_stroke_ratio: float = 0.35, font_scale: float = 1.0):
        self.original_pdf_path = original_pdf_path
        self.output_pdf_path = output_pdf_path or (original_pdf_path.replace(".pdf", "_translated.pdf") if original_pdf_path else None)
        self.ink_thresh = int(ink_thresh)
        self.dilate_iter = int(dilate_iter)
        self.max_stroke_ratio = float(max_stroke_ratio)
        self.font_scale = float(font_scale)

    # ── Page rendering ──────────────────────────────────────────────────────
    # Heavy CPU work (mask building, Telea, text drawing, JPEG encode) runs in worker
    # threads so the event loop (SSE progress, extraction/translation stages) stays
    # responsive (BUG.md B3). Only rasterization stays on the calling thread because
    # the fitz.Page / Document is shared and PyMuPDF is not thread-safe.

    @staticmethod
    def _rasterize(src_page):
        TARGET_HEIGHT = 1600.0
        scale = min(2.0, TARGET_HEIGHT / max(1.0, float(src_page.rect.height)))
        pix = src_page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img, scale

    def _build_masks(self, img, scale, page_data, translated_text_map):
        """Build Telea / LaMa masks and the list of blocks to draw (thread-safe, pure numpy)."""
        page_num = page_data["page_num"]
        img_np = np.array(img)
        gray_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        telea_mask_np = np.zeros((img.height, img.width), dtype=np.uint8)
        lama_mask_np = np.zeros((img.height, img.width), dtype=np.uint8)
        has_telea_masks = False
        has_lama_masks = False
        blocks_to_render = []

        for block in page_data["blocks"]:
            block_id = block["id"]
            translated = translated_text_map.get((page_num, block_id), "").strip()
            if not translated:
                continue

            bx0, by0, bx1, by1 = block["bbox"]
            px0 = max(0, min(img.width - 1, int(bx0 * scale)))
            py0 = max(0, min(img.height - 1, int(by0 * scale)))
            px1 = max(0, min(img.width, int(bx1 * scale)))
            py1 = max(0, min(img.height, int(by1 * scale)))
            if px1 <= px0 or py1 <= py0:
                continue

            crop_rgb = img_np[py0:py1, px0:px1]
            mask_res = _create_stroke_mask(
                crop_rgb,
                ink_thresh=self.ink_thresh,
                dilate_iter=self.dilate_iter,
                max_stroke_ratio=self.max_stroke_ratio,
                textured=_ring_is_textured(gray_np, px0, py0, px1, py1)
            )
            stroke_mask = mask_res[0]
            bg_val = mask_res[1]
            is_textured = mask_res[2] if len(mask_res) > 2 else (bg_val < 170)

            # Smart Routing:
            # - Clean/white bubble: Telea (<15ms, perfectly sharp, zero downsample blur)
            # - Textured/screentone/dark scene: LaMa neural inpainting
            if is_textured:
                lama_mask_np[py0:py1, px0:px1] = np.maximum(lama_mask_np[py0:py1, px0:px1], stroke_mask)
                has_lama_masks = True
            else:
                telea_mask_np[py0:py1, px0:px1] = np.maximum(telea_mask_np[py0:py1, px0:px1], stroke_mask)
                has_telea_masks = True

            blocks_to_render.append((px0, py0, px1, py1, translated, block))

        return {
            "blocks": blocks_to_render,
            "telea_mask": telea_mask_np if has_telea_masks else None,
            "lama_mask": lama_mask_np if has_lama_masks else None,
        }

    def _draw_translations(self, healed, blocks_to_render):
        """Draw translated CJK text into `healed` in place (shared by page render and preview)."""
        with _DRAW_LOCK:
            draw = ImageDraw.Draw(healed)
            for (px0, py0, px1, py1, translated, block) in blocks_to_render:
                bw = max(px1 - px0, 1)
                bh = max(py1 - py0, 1)

                orig_color = block.get("color", None)
                if orig_color and isinstance(orig_color, (list, tuple)) and len(orig_color) == 3:
                    r = int(orig_color[0] * 255)
                    g = int(orig_color[1] * 255)
                    b_ch = int(orig_color[2] * 255)
                    lum = 0.299 * r + 0.587 * g + 0.114 * b_ch
                    if lum > 230:
                        bg = _sample_bg(healed, px0, py0, px1, py1)
                        fg = _contrast_color(bg)
                    else:
                        fg = (r, g, b_ch)
                else:
                    bg = _sample_bg(healed, px0, py0, px1, py1)
                    fg = _contrast_color(bg)

                font, lines, fs = _best_font(translated, bw, bh, font_scale=self.font_scale)
                lh = fs * 1.25
                total_h = len(lines) * lh
                curr_y = py0 + (bh - total_h) / 2

                for line in lines:
                    try:
                        lw = font.getbbox(line)[2] - font.getbbox(line)[0]
                    except Exception:
                        lw = len(line) * fs * 0.9
                    curr_x = px0 + (bw - lw) / 2

                    if isinstance(fg, tuple) and len(fg) == 3:
                        fg_lum = 0.299 * fg[0] + 0.587 * fg[1] + 0.114 * fg[2]
                    else:
                        fg_lum = 255 if fg == "white" else 0

                    st_fill = "black" if fg_lum > 127 else "white"
                    st_width = max(1, int(fs / 16))

                    draw.text(
                        (curr_x, curr_y),
                        line,
                        fill=fg,
                        font=font,
                        stroke_width=st_width,
                        stroke_fill=st_fill
                    )
                    curr_y += lh
        return healed

    async def render_single_page_to_temp(self, page_data, src_page, translated_text_map, temp_dir, sem):
        """
        Renders a single page using high-fidelity inpainting and CJK text layout, saving it to a temp JPEG file.
        Returns the path to the temp JPEG file.
        """
        page_num = page_data["page_num"]
        t_page = time.time()
        t = time.time()
        img, scale = self._rasterize(src_page)
        _rstat("rasterize", time.time() - t)
        t = time.time()
        prep = await asyncio.to_thread(self._build_masks, img, scale, page_data, translated_text_map)
        _rstat("masks", time.time() - t)
        temp_path = os.path.join(temp_dir, f"page_{page_num}.jpg")

        if not prep["blocks"]:
            await asyncio.to_thread(_save_jpeg, img, temp_path)
            _rstat("page_total", time.time() - t_page)
            return temp_path

        healed = img
        if prep["telea_mask"] is not None:
            t = time.time()
            healed = await asyncio.to_thread(_fast_telea_inpaint, healed, Image.fromarray(prep["telea_mask"]))
            _rstat("telea", time.time() - t)
        if prep["lama_mask"] is not None:
            # Neural LaMa inpainting only for textured / dark regions (bounded by sem)
            t = time.time()
            async with sem:
                healed = await asyncio.to_thread(_lama_inpaint, healed, Image.fromarray(prep["lama_mask"]))
            _rstat("lama_page", time.time() - t)

        def _finish():
            self._draw_translations(healed, prep["blocks"])
            _save_jpeg(healed, temp_path)

        t = time.time()
        await asyncio.to_thread(_finish)
        _rstat("draw_save", time.time() - t)
        _rstat("page_total", time.time() - t_page)
        return temp_path

    def render_preview_images(self, page_data, src_page, translated_text_map):
        """
        Renders preview images for interactive sidebar tuning:
        Returns base64 data URLs for:
        - original: high-res original page
        - mask: translucent magenta mask overlay on original page
        - result: inpainted and translated CJK text overlay
        """
        import io
        import base64

        TARGET_HEIGHT = 1600.0
        SCALE = min(2.0, TARGET_HEIGHT / max(1.0, float(src_page.rect.height)))

        # 1. Render page to image
        mat = fitz.Matrix(SCALE, SCALE)
        pix = src_page.get_pixmap(matrix=mat, alpha=False)
        orig_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_np = np.array(orig_img)
        gray_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # 2. Build precision mask
        mask_np = np.zeros((orig_img.height, orig_img.width), dtype=np.uint8)
        blocks_to_render = []
        page_num = page_data.get("page_num", 1)

        for block in page_data.get("blocks", []):
            block_id = block["id"]
            translated = translated_text_map.get((page_num, block_id), "").strip()
            # If no translation in map yet, use block's translated text or original text as preview placeholder
            if not translated:
                translated = block.get("translated_text", "").strip() or block.get("text", "").strip()
            if not translated:
                continue

            bx0, by0, bx1, by1 = block["bbox"]
            px0 = max(0, min(orig_img.width - 1, int(bx0 * SCALE)))
            py0 = max(0, min(orig_img.height - 1, int(by0 * SCALE)))
            px1 = max(0, min(orig_img.width, int(bx1 * SCALE)))
            py1 = max(0, min(orig_img.height, int(by1 * SCALE)))

            if px1 <= px0 or py1 <= py0:
                continue

            crop_rgb = img_np[py0:py1, px0:px1]
            mask_res = _create_stroke_mask(
                crop_rgb,
                ink_thresh=self.ink_thresh,
                dilate_iter=self.dilate_iter,
                max_stroke_ratio=self.max_stroke_ratio,
                textured=_ring_is_textured(gray_np, px0, py0, px1, py1)
            )
            stroke_mask = mask_res[0]
            bg_val = mask_res[1]
            is_textured = mask_res[2] if len(mask_res) > 2 else (bg_val < 170)
            
            mask_np[py0:py1, px0:px1] = np.maximum(mask_np[py0:py1, px0:px1], stroke_mask)
            blocks_to_render.append((px0, py0, px1, py1, translated, block))

        # 3. Create Mask Overlay visualization (Hot magenta / crimson highlight on original image)
        mask_vis = img_np.copy()
        mask_bool = mask_np > 0
        if np.any(mask_bool):
            # Blend original with translucent vivid crimson (RGB 255, 30, 90)
            overlay_color = np.array([255, 30, 90], dtype=np.float32)
            mask_vis[mask_bool] = (0.35 * mask_vis[mask_bool].astype(np.float32) + 0.65 * overlay_color).astype(np.uint8)
        mask_img = Image.fromarray(mask_vis)

        # 4. Inpaint with Smart Hybrid Inpainting
        if np.any(mask_bool):
            healed = _lama_inpaint(orig_img, Image.fromarray(mask_np))
        else:
            healed = Image.fromarray(img_np)

        # 5. Draw CJK text
        self._draw_translations(healed, blocks_to_render)

        def img_to_b64(pil_im, quality=80):
            buf = io.BytesIO()
            pil_im.save(buf, format="JPEG", quality=quality, optimize=True)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            "original": img_to_b64(orig_img, 75),
            "mask": img_to_b64(mask_img, 80),
            "result": img_to_b64(healed, 85),
            "blocks_count": len(blocks_to_render)
        }

    async def render_translated_pdf(self, layout_data, translated_text_map, progress_callback=None):
        """
        Renders PDF pages concurrently to a temp folder and assembles them in order.
        """
        import asyncio
        import tempfile
        import shutil

        src_doc = fitz.open(self.original_pdf_path)
        total_pages = len(layout_data)
        
        # Create a temp directory for page JPEGs
        temp_dir = tempfile.mkdtemp(prefix="pdf_render_")
        
        sem = asyncio.Semaphore(1)
        progress_count = 0
        progress_lock = asyncio.Lock()
        
        async def process_single_page(page_data):
            nonlocal progress_count
            page_num = page_data["page_num"]
            src_page = src_doc[page_num - 1]
            
            temp_path = await self.render_single_page_to_temp(
                page_data=page_data,
                src_page=src_page,
                translated_text_map=translated_text_map,
                temp_dir=temp_dir,
                sem=sem
            )
            
            async with progress_lock:
                progress_count += 1
                if progress_callback:
                    progress_callback(progress_count, total_pages)
            
            return page_num, temp_path

        # Run all pages concurrently
        tasks = [process_single_page(pd) for pd in layout_data]
        results = await asyncio.gather(*tasks)
        temp_paths = dict(results)

        # Assemble the final PDF in order
        out_doc = fitz.open()
        for page_num in range(1, total_pages + 1):
            src_page = src_doc[page_num - 1]
            out_page = out_doc.new_page(
                width=src_page.rect.width,
                height=src_page.rect.height
            )
            if page_num in temp_paths and os.path.exists(temp_paths[page_num]):
                out_page.insert_image(out_page.rect, filename=temp_paths[page_num])

        src_doc.close()
        out_doc.save(self.output_pdf_path, garbage=4, deflate=True)
        out_doc.close()
        
        # Clean up temp files
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        print(f"[✓] LaMa Render Complete: {self.output_pdf_path}")
        return self.output_pdf_path


def _embed_image_on_page(page: fitz.Page, img: Image.Image):
    """Insert a PIL image to fill an entire fitz page using JPEG to save space and time."""
    import io
    buf = io.BytesIO()
    # Convert to RGB if it has an alpha channel, JPEG doesn't support RGBA
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buf.seek(0)
    img.save(buf, format="JPEG", quality=80, optimize=True)
    buf.seek(0)
    page.insert_image(page.rect, stream=buf.read())
