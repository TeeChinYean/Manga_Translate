#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fonts the translation can be lettered with.

The list is: fonts dropped into `pdf_translate/fonts/`, MANGA_FONT, and well-known system CJK fonts that
really contain Simplified Chinese glyphs. Clients only ever send an id from this list, never a path.
Id "" means "automatic" (the renderer's default choice).
"""
import os
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

FONTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts")
FONT_EXTS = (".ttf", ".otf", ".ttc")
COVERAGE_SAMPLE = "汉字的是们这说你我他"      # Simplified Chinese; a font missing them would draw empty boxes

# (file name, label) -> looked up in the system font folders below
SYSTEM_FONTS = [
    ("msyh.ttc", "微软雅黑"), ("msyhbd.ttc", "微软雅黑 粗体"), ("simhei.ttf", "黑体"), ("simsun.ttc", "宋体 / 新宋体"),
    ("simkai.ttf", "楷体"), ("simfang.ttf", "仿宋"), ("Deng.ttf", "等线"), ("Dengb.ttf", "等线 粗体"),
    ("STXIHEI.TTF", "华文细黑"), ("STKAITI.TTF", "华文楷体"), ("STSONG.TTF", "华文宋体"), ("STFANGSO.TTF", "华文仿宋"),
    ("FZSTK.TTF", "方正舒体"), ("SIMLI.TTF", "隶书"), ("SIMYOU.TTF", "幼圆"),
    ("NotoSansCJK-Regular.ttc", "思源黑体 Regular"), ("NotoSansCJK-Medium.ttc", "思源黑体 Medium"),
    ("NotoSansCJK-Bold.ttc", "思源黑体 Bold"), ("NotoSerifCJK-Regular.ttc", "思源宋体 Regular"),
    ("NotoSerifCJK-Bold.ttc", "思源宋体 Bold"), ("PingFang.ttc", "苹方"), ("Songti.ttc", "宋体-简"),
    ("STHeiti Medium.ttc", "华文黑体"), ("Hiragino Sans GB.ttc", "冬青黑体"),
    ("wqy-microhei.ttc", "文泉驿微米黑"), ("wqy-zenhei.ttc", "文泉驿正黑"),
]
SYSTEM_DIRS = [
    "C:\\Windows\\Fonts", os.path.expanduser("~\\AppData\\Local\\Microsoft\\Windows\\Fonts"),
    "/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"), os.path.expanduser("~/.local/share/fonts"),
    "/System/Library/Fonts", "/Library/Fonts", os.path.expanduser("~/Library/Fonts"),
]


@lru_cache(maxsize=256)
def covers_chinese(path: str) -> bool:
    """True when the font has real glyphs for Simplified Chinese (not the empty 'tofu' box)."""
    try:
        font = ImageFont.truetype(path, 28)
    except Exception:
        return False

    def bitmap(ch):
        img = Image.new("L", (48, 48), 0)
        ImageDraw.Draw(img).text((4, 4), ch, font=font, fill=255)
        return img.tobytes()

    try:
        tofu = bitmap("\U0010FFFD")
        blank = bytes(48 * 48)
        return all(bitmap(ch) not in (tofu, blank) for ch in COVERAGE_SAMPLE)
    except Exception:
        return False


def _find_system(filename: str):
    for base in SYSTEM_DIRS:
        direct = os.path.join(base, filename)
        if os.path.isfile(direct):
            return direct
    lower = filename.lower()
    for base in SYSTEM_DIRS[2:6]:          # Linux: fonts live in nested folders
        for root, _dirs, files in os.walk(base):
            for f in files:
                if f.lower() == lower:
                    return os.path.join(root, f)
    return None


@lru_cache(maxsize=1)
def _system_catalog() -> tuple:
    out = []
    for filename, label in SYSTEM_FONTS:
        path = _find_system(filename)
        if path and covers_chinese(path):
            out.append({"id": "sys:" + filename, "name": label, "path": path})
    return tuple(out)


def catalog() -> list:
    """[{"id", "name", "path"}] of the usable fonts (the folder is re-read every call, so new files show up)."""
    out, seen = [], set()

    def add(item):
        if item["id"] not in seen:
            seen.add(item["id"])
            out.append(item)

    try:
        for f in sorted(os.listdir(FONTS_DIR)):
            p = os.path.join(FONTS_DIR, f)
            if f.lower().endswith(FONT_EXTS) and os.path.isfile(p) and covers_chinese(p):
                add({"id": "file:" + f, "name": os.path.splitext(f)[0] + "（fonts 文件夹）", "path": p})
    except OSError:
        pass
    env = os.getenv("MANGA_FONT")
    if env and os.path.isfile(env) and covers_chinese(env):
        add({"id": "env", "name": os.path.basename(env) + "（MANGA_FONT）", "path": env})
    for item in _system_catalog():
        add(dict(item))
    return out


def list_fonts() -> list:
    """What the UI shows: automatic first, then the catalog (no paths)."""
    return [{"id": "", "name": "自动（默认）"}] + [{"id": c["id"], "name": c["name"]} for c in catalog()]


def resolve_font(font_id) -> str:
    """Path of a font id from the catalog, or "" (automatic) for '' / unknown ids."""
    if not font_id or not isinstance(font_id, str):
        return ""
    for c in catalog():
        if c["id"] == font_id:
            return c["path"]
    return ""


def normalize_font_id(font_id) -> str:
    """The id itself when it is in the catalog, else '' (automatic)."""
    return font_id if resolve_font(font_id) else ""
