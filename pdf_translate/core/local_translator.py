#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ Local Offline Translation Module
Uses Helsinki-NLP/opus-mt-en-zh via HuggingFace Transformers.
- No hallucinations (pure seq2seq translation, not chat LLM)
- No internet needed after first model download (~300MB, cached)
- No "对不起我无法..." chat pollution
- Batch translation for efficiency
"""

import re
import logging
import threading

logger = logging.getLogger(__name__)

# Proper nouns come from the active manga series' term file (core.engine, data/terms/),
# so this translator no longer carries one manga's names for every book.

# OCR garbage patterns that should NOT be translated (return empty string)
_NOISE_PATTERNS = [
    re.compile(r'^[A-Z]{2,5}\s*[A-Z]{0,5}$'),          # All-caps short strings like "WHIPP", "4T HVP"
    re.compile(r'^\d+\s*[A-Z]{1,3}$'),                  # Like "4T"
    re.compile(r'^[^a-zA-Z\u4e00-\u9fff]{3,}$'),        # Only symbols/numbers
    re.compile(r'^[\W\d_]+$'),                           # Only non-word chars
]

_NOISE_KEYWORDS = {
    "erp", "hvp", "whipp", "mrnch", "krngh", "yaaawn", "shhh", "hmmm",
    "argh", "ugh", "oof", "urk", "eep", "gah", "bam", "pow", "zap",
    "wham", "thud", "crash", "bang", "boom", "ka-", "ka ", "pa-", "ta-",
}


def _is_noise(text: str) -> bool:
    """Return True if text is OCR garbage that should be skipped."""
    t = text.strip()
    if not t or len(t) < 2:
        return True
    tl = t.lower()
    # Short all-caps garbage
    if t.isupper() and len(t) <= 6 and not any(c.isspace() for c in t):
        return True
    # Keyword match
    if any(kw in tl for kw in _NOISE_KEYWORDS):
        return True
    # Pattern match
    for pat in _NOISE_PATTERNS:
        if pat.match(t):
            return True
    # Nearly no alphabetic content
    alpha = [c for c in t if c.isalpha()]
    if alpha and len(alpha) / len(t) < 0.4:
        return True
    return False


def _apply_proper_nouns(text: str) -> str:
    """Replace the active series' curated proper nouns after translation."""
    try:
        from core import engine as _engine
        terms = dict(_engine._PROPER_NOUNS)
        replace = _engine._replace_term
    except Exception:
        return text
    for src, tgt in terms.items():
        text = replace(text, src, tgt)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Singleton translator (lazy-loaded on first use)
# ─────────────────────────────────────────────────────────────────────────────

_translator_instance = None
_translator_lock = threading.Lock()


def _get_translator():
    global _translator_instance
    if _translator_instance is not None:
        return _translator_instance
    with _translator_lock:
        if _translator_instance is not None:
            return _translator_instance
        try:
            from transformers import pipeline, MarianMTModel, MarianTokenizer
            logger.info("[LocalTranslator] Loading Helsinki-NLP/opus-mt-en-zh model...")
            tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
            model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
            model.eval()
            _translator_instance = (tokenizer, model)
            logger.info("[LocalTranslator] ✅ opus-mt-en-zh model loaded and ready.")
        except Exception as e:
            logger.error(f"[LocalTranslator] Failed to load opus-mt-en-zh: {e}")
            _translator_instance = None
    return _translator_instance


class LocalTranslator:
    """
    Thin wrapper around Helsinki-NLP/opus-mt-en-zh for manga dialogue translation.
    Handles batch translation and proper noun post-processing.
    """

    def translate_one(self, text: str) -> str:
        """Translate a single string. Returns empty string for noise."""
        results = self.translate_batch([text])
        return results[0]

    def translate_batch(self, texts: list) -> list:
        """
        Translate a list of strings.
        Returns list of translated strings (same length as input).
        Noise/garbage → empty string.
        """
        if not texts:
            return []

        results = [""] * len(texts)
        to_translate = []  # (original_index, text)

        for i, text in enumerate(texts):
            t = text.strip()
            if not t or _is_noise(t):
                results[i] = ""
                continue
            to_translate.append((i, t))

        if not to_translate:
            return results

        translator = _get_translator()
        if translator is None:
            logger.warning("[LocalTranslator] Model not available, returning empty.")
            return results

        tokenizer, model = translator
        try:
            import torch
            batch_texts = [txt for _, txt in to_translate]

            # Tokenize in batch
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )

            with torch.no_grad():
                translated_tokens = model.generate(
                    **inputs,
                    num_beams=4,
                    max_length=256,
                    early_stopping=True,
                )

            translated_texts = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

            for (orig_idx, _), translated in zip(to_translate, translated_texts):
                clean = translated.strip()
                # Post-process: apply proper nouns
                clean = _apply_proper_nouns(clean)
                # Reject if model returned the original English unchanged
                orig_text = texts[orig_idx].strip()
                has_chinese = any('\u4e00' <= c <= '\u9fff' for c in clean)
                if not has_chinese:
                    clean = ""
                results[orig_idx] = clean

        except Exception as e:
            logger.error(f"[LocalTranslator] Batch translation error: {e}")

        return results


# Module-level singleton
_local_translator = None
_lt_lock = threading.Lock()


def get_local_translator() -> LocalTranslator:
    global _local_translator
    if _local_translator is None:
        with _lt_lock:
            if _local_translator is None:
                _local_translator = LocalTranslator()
    return _local_translator
