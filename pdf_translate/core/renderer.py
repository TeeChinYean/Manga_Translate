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
import numpy as np
import logging
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

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

# ── LaMa model (loaded lazily once) ───────────────────────────────────────────
_LAMA_INPAINTER = None
_MIT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "manga_translator_source",
    "manga-image-translator-main"
)

async def _get_lama():
    global _LAMA_INPAINTER
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if _LAMA_INPAINTER is not None:
        if hasattr(_LAMA_INPAINTER, 'model') and _LAMA_INPAINTER.model is not None:
            _LAMA_INPAINTER.model = _LAMA_INPAINTER.model.to(device)
        return _LAMA_INPAINTER

    try:
        sys.path.insert(0, os.path.abspath(_MIT_PATH))
        from manga_translator.inpainting import get_inpainter
        
        if device == "cuda":
            logger.info("⚡ [Renderer] Running LaMa inpainting on NVIDIA GPU (CUDA).")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = "xpu"
            logger.info("⚡ [Renderer] Offloading LaMa inpainting to Intel iGPU (XPU).")
        elif hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            logger.info("⚡ [Renderer] Offloading LaMa inpainting to Apple Silicon (MPS).")
        else:
            device = "cpu"
            logger.info("⚡ [Renderer] Running LaMa inpainting on CPU.")

        inpainter = get_inpainter("lama_large")
        
        # Native async load without event loop conflict
        await inpainter.load(device)
        
        _LAMA_INPAINTER = inpainter
        logger.info(f"[✓] LaMa Large inpainting model loaded natively on {device}.")
        print(f"[✓] LaMa Large inpainting model ready on {device}.")
    except Exception as e:
        logger.warning(f"[!] LaMa load failed natively ({e}), falling back to PIL background-sampling.")
        _LAMA_INPAINTER = None
    return _LAMA_INPAINTER

def unload_models():
    """Moves PyTorch LaMa models from VRAM to CPU RAM (shared memory) for instant reload later."""
    global _LAMA_INPAINTER
    if _LAMA_INPAINTER is not None:
        if hasattr(_LAMA_INPAINTER, 'model') and _LAMA_INPAINTER.model is not None:
            _LAMA_INPAINTER.model = _LAMA_INPAINTER.model.to('cpu')
            
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

def _best_font(text: str, box_w: int, box_h: int):
    # Calculate usable area with 15% padding to prevent text from touching the borders
    pad_w = int(box_w * 0.15)
    pad_h = int(box_h * 0.15)
    usable_w = max(box_w - pad_w, 20)
    usable_h = max(box_h - pad_h, 20)
    
    text_len = max(len(text), 1)
    
    # Estimate max font size based on area (fs * fs * 1.25 * len <= w * h)
    import math
    estimated_max_fs = int(math.sqrt((usable_w * usable_h) / (text_len * 1.25)))
    
    # Cap the theoretical max size by the box dimensions
    max_fs = min(estimated_max_fs, usable_w, usable_h)
    
    # Add a buffer to search slightly above the estimate
    max_fs = int(max_fs * 1.5)
    max_fs = min(max_fs, 200) # Hard limit
    min_fs = 10
    
    best_font, best_lines, best_fs = None, [], min_fs
    
    # Iterate downwards to find the first font size that fits the height
    for fs in range(max_fs, min_fs - 1, -1):
        font = _get_font(fs)
        lines = _wrap_cjk(text, font, usable_w)
        line_height = fs * 1.25
        total_h = len(lines) * line_height
        
        if total_h <= usable_h:
            # Verify no single line exceeds usable_w due to unbreakable words
            max_line_w = 0
            for line in lines:
                try:
                    lw = font.getbbox(line)[2] - font.getbbox(line)[0]
                except Exception:
                    lw = len(line) * fs
                max_line_w = max(max_line_w, lw)
                
            if max_line_w <= usable_w + 5: # Allow tiny overflow
                return font, lines, fs
            
    # Fallback to the minimum font size if nothing fits nicely
    font = _get_font(min_fs)
    return font, _wrap_cjk(text, font, usable_w), min_fs


async def _lama_inpaint_image(img_pil: Image.Image, mask_pil: Image.Image) -> Image.Image:
    """Run LaMa inpainting with correct InpainterConfig parameters natively."""
    inpainter = await _get_lama()
    if inpainter is None:
        return _fallback_fill(img_pil, mask_pil)
    try:
        sys.path.insert(0, os.path.abspath(_MIT_PATH))
        from manga_translator.config import InpainterConfig
        config = InpainterConfig()
        
        img_np = np.array(img_pil.convert("RGB"))
        mask_np = np.array(mask_pil.convert("L"))

        # Optimize inpainting resolution to 720 to reduce VRAM load and accelerate processing
        result = await inpainter.inpaint(img_np, mask_np, config, 720, False)

        if isinstance(result, np.ndarray):
            return Image.fromarray(result.astype(np.uint8))
        return img_pil
    except Exception as e:
        logger.warning(f"[!] LaMa inpainting error: {e}. Falling back to border fill.")
        return _fallback_fill(img_pil, mask_pil)


def _fallback_fill(img_pil: Image.Image, mask_pil: Image.Image) -> Image.Image:
    """Fallback: fill masked regions with sampled surrounding color safely."""
    img = img_pil.copy()
    mask_np = np.array(mask_pil.convert("L"))
    
    # Use cv2 to find individual mask regions so we don't draw one giant rectangle
    import cv2
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    draw = ImageDraw.Draw(img)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 2 or h < 2:
            continue
        x0, y0, x1, y1 = x, y, x + w, y + h
        bg = _sample_bg(img, x0, y0, x1, y1)
        draw.rectangle([x0, y0, x1, y1], fill=bg)
        
    return img


# ── Main Renderer Class ────────────────────────────────────────────────────────
class PDFLayoutRenderer:
    def __init__(self, original_pdf_path, output_pdf_path=None):
        self.original_pdf_path = original_pdf_path
        self.output_pdf_path = output_pdf_path or original_pdf_path.replace(".pdf", "_translated.pdf")

    async def render_single_page_to_temp(self, page_data, src_page, translated_text_map, temp_dir, sem):
        """
        Renders a single page using LaMa / CJK text layout, saving it to a temp JPEG file.
        Returns the path to the temp JPEG file.
        """
        page_num = page_data["page_num"]
        TARGET_HEIGHT = 1600.0
        SCALE = min(2.0, TARGET_HEIGHT / max(1.0, float(src_page.rect.height)))
        
        # 1. Render page to image
        mat = fitz.Matrix(SCALE, SCALE)
        pix = src_page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # 2. Build mask for blocks that have translations
        mask = Image.new("L", img.size, 0)
        mask_draw = ImageDraw.Draw(mask)

        blocks_to_render = []
        for block in page_data["blocks"]:
            block_id = block["id"]
            translated = translated_text_map.get((page_num, block_id), "").strip()
            if not translated:
                continue

            bx0, by0, bx1, by1 = block["bbox"]
            px0 = int(bx0 * SCALE)
            py0 = int(by0 * SCALE)
            px1 = int(bx1 * SCALE)
            py1 = int(by1 * SCALE)
            pad = 3
            mask_draw.rectangle(
                [px0 - pad, py0 - pad, px1 + pad, py1 + pad],
                fill=255
            )
            blocks_to_render.append((px0, py0, px1, py1, translated, block))

        temp_path = os.path.join(temp_dir, f"page_{page_num}.jpg")

        if not blocks_to_render:
            img.save(temp_path, format="JPEG", quality=80, optimize=True)
        else:
            async with sem:
                healed = await _lama_inpaint_image(img, mask)

            # Draw CJK text
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

                font, lines, fs = _best_font(translated, bw, bh)
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
            
            healed.save(temp_path, format="JPEG", quality=80, optimize=True)
            
        return temp_path

    async def render_translated_pdf(self, layout_data, translated_text_map, progress_callback=None):
        """
        Renders PDF pages concurrently to a temp folder and assembles them in order.
        """
        import asyncio
        import tempfile
        import shutil

        # Warm up the LaMa model natively
        await _get_lama()

        src_doc = fitz.open(self.original_pdf_path)
        total_pages = len(layout_data)
        
        # Create a temp directory for page JPEGs
        temp_dir = tempfile.mkdtemp(prefix="pdf_render_")
        
        sem = asyncio.Semaphore(3)
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
