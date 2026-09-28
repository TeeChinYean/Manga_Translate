#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ ANTIGRAVITY - Translation Engine v7
Primary:  Google Translate API (30 concurrent workers, zero GPU, zero hallucinations)
Fallback: Helsinki-NLP/opus-mt-en-zh local offline model (when Google fails)
Final:    Qwen local LLM (when both above fail, strictly constrained prompt)
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

import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

import json
import os

logger = logging.getLogger(__name__)

# ── Turbovec LLM Configuration (Replaces Ollama) ──────────────────────────────
TURBOVEC_API_URL = os.getenv("TURBOVEC_API_URL", "http://localhost:18088/v1/chat/completions")
LLAMA_SERVER_DIRECT_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:18089/v1/chat/completions")
TURBOVEC_MODEL = os.getenv("TURBOVEC_MODEL", "docker.io/ai/qwen3.5:4b-q4_K_M")

_LLAMA_KEY = None

def _get_llama_api_key():
    global _LLAMA_KEY
    if _LLAMA_KEY is None:
        key_env = os.getenv("LLAMA_API_KEY")
        if key_env:
            _LLAMA_KEY = key_env.strip()
        else:
            candidate_paths = [
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "qwen_turbovec_rag", "app", "storage", "llama_api_key.txt"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "..", "qwen_turbovec_rag", "app", "storage", "llama_api_key.txt"),
                r"c:\Users\Work\Desktop\project\qwen_turbovec_rag\app\storage\llama_api_key.txt"
            ]
            for p in candidate_paths:
                p_norm = os.path.normpath(p)
                if os.path.exists(p_norm):
                    try:
                        with open(p_norm, "r", encoding="utf-8") as f:
                            _LLAMA_KEY = f.read().strip()
                            if _LLAMA_KEY:
                                break
                    except Exception:
                        pass
    return _LLAMA_KEY

def ensure_turbovec_llm_ready(auto_launch: bool = True, max_wait_seconds: int = 25) -> bool:
    """
    Checks if Turbovec LLM (port 18089 direct or 18088 gateway) is active and pre-warmed.
    If not running and auto_launch=True, starts llm_launcher.py in the background and waits until healthy.
    """
    # 1. Quick probe
    for probe_url in ["http://127.0.0.1:18089/health", "http://127.0.0.1:18088/health"]:
        try:
            r = httpx.get(probe_url, timeout=1.5)
            if r.status_code == 200:
                logger.info(f"⚡ [Turbovec LLM] Engine already running and healthy ({probe_url}).")
                return True
        except Exception:
            pass

    if not auto_launch:
        return False

    # 2. Find llm_launcher.py in qwen_turbovec_rag
    candidate_dirs = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "qwen_turbovec_rag"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "..", "qwen_turbovec_rag"),
        r"c:\Users\Work\Desktop\project\qwen_turbovec_rag"
    ]
    rag_dir = None
    for d in candidate_dirs:
        norm_d = os.path.normpath(d)
        if os.path.exists(os.path.join(norm_d, "app", "llm_launcher.py")):
            rag_dir = norm_d
            break

    if not rag_dir:
        logger.warning("⚡ [Turbovec LLM] qwen_turbovec_rag not found; cannot auto-launch LLM.")
        return False

    logger.info(f"⚡ [Turbovec LLM] Starting local Qwen 3.5 4B model via {rag_dir}...")
    import subprocess
    cmd = [sys.executable, os.path.join("app", "llm_launcher.py"), "--model", "1"]
    try:
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        DETACHED_PROCESS = 0x00000008
        subprocess.Popen(
            cmd,
            cwd=rag_dir,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
            close_fds=True
        )
    except Exception as e:
        logger.warning(f"⚡ [Turbovec LLM] Failed spawning llm_launcher.py: {e}")
        return False

    # 3. Wait until port 18089 responds to /health
    start_t = time.time()
    while time.time() - start_t < max_wait_seconds:
        time.sleep(1.0)
        try:
            r = httpx.get("http://127.0.0.1:18089/health", timeout=1.5)
            if r.status_code == 200:
                elapsed = time.time() - start_t
                logger.info(f"✔ [Turbovec LLM] Engine booted and responsive in {elapsed:.1f}s!")
                # Send warm-up inference request
                try:
                    _call_turbovec_llm({
                        "model": TURBOVEC_MODEL,
                        "messages": [{"role": "user", "content": "1"}],
                        "max_tokens": 2
                    }, timeout=5.0)
                    logger.info("✔ [Turbovec LLM] Pre-warm inference successful!")
                except Exception:
                    pass
                return True
        except Exception:
            continue

    logger.warning(f"⚠️ [Turbovec LLM] Engine did not become healthy within {max_wait_seconds}s.")
    return False

# ── Per-batch timing breakdown (where does translation time go?) ─────────────
import threading as _threading
_TLS = _threading.local()


def _stats_begin():
    _TLS.stats = {}


def _stats_end() -> dict:
    st = getattr(_TLS, "stats", None) or {}
    _TLS.stats = None
    return st


def _stats_add(tag: str, seconds: float, completion_tokens: int = 0, ok: bool = True):
    st = getattr(_TLS, "stats", None)
    if st is None:
        return
    e = st.setdefault(tag, {"calls": 0, "seconds": 0.0, "completion_tokens": 0, "failed": 0})
    e["calls"] += 1
    e["seconds"] = round(e["seconds"] + seconds, 2)
    e["completion_tokens"] += int(completion_tokens or 0)
    if not ok:
        e["failed"] += 1


def merge_stats(total: dict, part: dict) -> dict:
    """Sum two timing breakdowns (used by main.py across pages)."""
    for tag, e in (part or {}).items():
        t = total.setdefault(tag, {"calls": 0, "seconds": 0.0, "completion_tokens": 0, "failed": 0})
        for k in ("calls", "completion_tokens", "failed"):
            t[k] += e.get(k, 0)
        t["seconds"] = round(t["seconds"] + e.get("seconds", 0.0), 2)
    return total


def _call_turbovec_llm(payload: dict, timeout: float = 15.0, tag: str = "llm"):
    t0 = time.time()
    data = _call_turbovec_llm_raw(payload, timeout=timeout)
    tokens = 0
    try:
        tokens = int(((data or {}).get("usage") or {}).get("completion_tokens") or 0)
    except Exception:
        pass
    _stats_add(tag, time.time() - t0, tokens, ok=data is not None)
    return data


def _stream_chat(url: str, payload: dict, headers: dict, timeout: float):
    """
    POST a streaming chat completion and assemble it into a normal (non-stream) response.

    Why streaming: llama-server runs one slot (-np 1). With a plain request, giving up on
    the client side does NOT stop generation, so every later request queues behind the
    runaway. When a stream is closed the server aborts the generation and frees the slot.
    `timeout` is a total deadline; on expiry the stream is closed and TimeoutError raised.
    Returns (status_code, data_or_None).
    """
    body = dict(payload, stream=True, stream_options={"include_usage": True})
    deadline = time.time() + timeout
    parts, usage, n_chunks, finish = [], None, 0, None
    read_timeout = max(1.0, min(30.0, timeout))
    with httpx.stream("POST", url, json=body, headers=headers,
                      timeout=httpx.Timeout(read_timeout, connect=5.0)) as r:
        if r.status_code != 200:
            try:
                r.read()
                text = r.text[:100]
            except Exception:
                text = ""
            logger.warning(f"[Turbovec LLM] HTTP {r.status_code} from {url}: {text}")
            return r.status_code, None
        for line in r.iter_lines():
            if time.time() > deadline:
                raise TimeoutError(f"LLM stream exceeded {timeout:.0f}s (closed, server slot freed)")
            if not line or not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except ValueError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                delta = ch.get("delta") or {}
                if delta.get("content"):
                    parts.append(delta["content"])
                    n_chunks += 1
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    return 200, {
        "choices": [{"message": {"role": "assistant", "content": "".join(parts)}, "finish_reason": finish}],
        "usage": usage or {"completion_tokens": n_chunks},
    }


def _is_timeout(e: BaseException) -> bool:
    return isinstance(e, TimeoutError) or isinstance(e, getattr(httpx, "TimeoutException", ()))


def _call_turbovec_llm_raw(payload: dict, timeout: float = 15.0):
    """
    Calls Turbovec OpenAI-compatible API endpoint (direct llama-server 18089 or Turbovec gateway 18088).
    Streams the response (see _stream_chat) so a timeout really frees the single server slot.
    Retries only connection-level failures; timeouts and HTTP 400 are returned immediately.
    """
    if "model" not in payload or not payload["model"]:
        payload["model"] = TURBOVEC_MODEL

    headers = {}
    key = _get_llama_api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"

    # Prioritize 18089 direct llama-server for speed, fallback to 18088 gateway
    endpoints = [LLAMA_SERVER_DIRECT_URL, TURBOVEC_API_URL]

    last_err = None
    for attempt in range(1, 4):
        last_err = None
        for url in endpoints:
            try:
                status, data = _stream_chat(url, payload, headers, timeout)
                _TLS.last_status = status
                if status == 200:
                    return data
                if status == 400:
                    # Bad request (e.g. unsupported response_format): retrying will not help
                    return None
            except Exception as e:
                last_err = e
                if _is_timeout(e):
                    _TLS.last_status = "timeout"
                    logger.warning(f"[Turbovec LLM] {e} on {url}; not retrying")
                    return None
                continue

        # If connection refused on attempt 1, auto-heal and boot the LLM server
        if attempt == 1 and last_err and "10061" in str(last_err):
            logger.info("⚡ [Turbovec LLM] Target machine refused connection. Attempting auto-boot...")
            ensure_turbovec_llm_ready(auto_launch=True, max_wait_seconds=20)
        elif attempt < 3:
            time.sleep(1.0)
            continue

    logger.warning(f"[Turbovec LLM] All endpoints failed: {last_err}")
    return None

# ── Proper noun pre-fixes and glossary loaded from JSON ───────────────────────
# Two tiers (BUG.md B5):
#   _PROPER_NOUNS : curated terms (proper_nouns.json + glossary.json). Used for exact-match
#                   overrides, input/output canonicalization and prompt hints.
#   _AUTO_TERMS   : terms the LLM discovered at runtime (proper_nouns_auto.json). Unverified,
#                   so they are ONLY offered to the LLM as prompt hints, never force-applied.
# Every term passes _is_valid_term(); ASCII terms only match on word boundaries so that
# junk like "K" / "SO" / "5" can no longer rewrite unrelated text.
import threading
from functools import lru_cache

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_PROPER_NOUNS = {}
_AUTO_TERMS = {}
_PROPER_NOUNS_PATH = os.path.join(_DATA_DIR, "proper_nouns.json")
_AUTO_TERMS_PATH = os.path.join(_DATA_DIR, "proper_nouns_auto.json")
_GLOSSARY_PATH = os.path.join(_DATA_DIR, "glossary.json")
_TERMS_LOCK = threading.Lock()
# Off by default: measured ~10s per page (31% of translation time) for ~11 tokens of output.
AUTO_PROPER_NOUNS_ENABLED = os.getenv("AUTO_PROPER_NOUNS", "0") == "1"

_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]")


def _is_valid_term(key, value) -> bool:
    """Reject terms that would corrupt text when substring-applied (single letters, digits, no-ops)."""
    if not isinstance(key, str) or not isinstance(value, str):
        return False
    k, v = key.strip(), value.strip()
    if not k or not v or k == v or len(k) > 30 or len(v) > 30:
        return False
    if k.replace(" ", "").isdigit():
        return False
    if _CJK_RE.search(k):
        return len(k) >= 2
    letters = sum(c.isalpha() for c in k)
    return len(k) >= 3 and letters >= 3


@lru_cache(maxsize=4096)
def _term_pattern(key: str):
    """Regex for a term: ASCII words need word boundaries, CJK terms match as substrings."""
    if _CJK_RE.search(key):
        return re.compile(re.escape(key))
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(key) + r"(?![A-Za-z0-9])")


def _term_in_text(key: str, text: str, ignore_case: bool = False) -> bool:
    pat = _term_pattern(key)
    if ignore_case and not _CJK_RE.search(key):
        return re.search(pat.pattern, text, flags=re.IGNORECASE) is not None
    return pat.search(text) is not None


def _replace_term(text: str, key: str, value: str) -> str:
    return _term_pattern(key).sub(lambda _m: value, text)


def _load_json_dict(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning(f"Could not load {path}: {e}")
        return {}


def _load_terms():
    rejected = 0
    for k, v in _load_json_dict(_PROPER_NOUNS_PATH).items():
        if _is_valid_term(k, v):
            _PROPER_NOUNS[k.strip()] = v.strip()
        else:
            rejected += 1
    try:
        if os.path.exists(_GLOSSARY_PATH):
            with open(_GLOSSARY_PATH, "r", encoding="utf-8") as f:
                gloss_data = json.load(f)
            if isinstance(gloss_data, list):
                for item in gloss_data:
                    src, tgt = item.get("src"), item.get("tgt")
                    if _is_valid_term(src, tgt) and src not in _PROPER_NOUNS:
                        _PROPER_NOUNS[src.strip()] = tgt.strip()
    except Exception as e:
        logger.warning(f"Could not load glossary: {e}")
    for k, v in _load_json_dict(_AUTO_TERMS_PATH).items():
        if _is_valid_term(k, v) and k not in _PROPER_NOUNS:
            _AUTO_TERMS[k.strip()] = v.strip()
    if rejected:
        logger.warning(f"[Terms] Ignored {rejected} invalid entries in proper_nouns.json (single letters, digits, no-op mappings).")


_load_terms()


def _get_relevant_glossary(texts: list) -> dict:
    """Finds curated and auto-discovered terms that occur in the current batch (prompt hints only)."""
    if not _PROPER_NOUNS and not _AUTO_TERMS:
        return {}
    combined = " ".join(str(t) for t in texts if t)
    matched = {}
    for source in (_AUTO_TERMS, _PROPER_NOUNS):  # curated wins on conflicts
        for k, v in list(source.items()):
            if _term_in_text(k, combined, ignore_case=True):
                matched[k] = v
    return matched

# ── Post-translation cleanup ─────────────────────────────────────────────────
_PAREN_EN_RE = re.compile(r'\s*\([A-Za-z][A-Za-z\s\-]*\)')

def _clean_input(text: str) -> str:
    text = " ".join(text.replace("\n", " ").split()).strip()
    
    # OCR often misreads 'a' as '@' (e.g., 'across' -> '@cross')
    text = text.replace("@", "a")
    
    for wrong, right in _PROPER_NOUNS.items():
        text = _replace_term(text, wrong, right)
        
    # Strip standalone words that are purely consonants (>=3 chars), often merged SFX like KRNCH, SHRF, TMP
    words = text.split()
    clean_words = []
    vowels = set("aeiouAEIOU")
    for w in words:
        w_alpha = "".join(c for c in w if c.isalpha())
        if len(w_alpha) >= 3 and not any(c in vowels for c in w_alpha):
            # Do not filter if it's CJK (Japanese doesn't use English vowels)
            has_cjk = any('\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' for c in w_alpha)
            if not has_cjk:
                # Skip this word (it's a consonant-only SFX or OCR noise)
                continue
        # Skip hallucinated OCR acronyms that Google translates literally into weird Chinese terms
        # Including user-reported background noise from specific manga pages
        if w_alpha.upper() in {"OOM", "ERP", "EAN", "TMP", "TM", "PDF", "ICTT", "ITI", "IKNCH", "NLALUIL", "NLAQUAL", "QIC", "LOVAIY"}:
            continue
        clean_words.append(w)
        
    return " ".join(clean_words).strip()

def _clean_output(text: str) -> str:
    if not text:
        return ""
    # Normalise "..." / "。。。" to "……" first, so the trailing-period strip below keeps ellipses
    text = _ELLIPSIS_RE.sub("……", text)
        
    # Strip hallucinated translation notes from the model (e.g., "注：这里的语气词...")
    import re
    text = re.sub(r'[(（]?(?:注|译注|翻译注)[：:].*', '', text, flags=re.DOTALL).strip()
        
    # Remove trailing English in parentheses like "(Kafna)"
    text = _PAREN_EN_RE.sub("", text).strip()
    
    # Replace internal semicolons with commas (standard in manga) and remove stray @ symbols
    text = text.replace("；", "，").replace(";", "，").replace("@", "")
    
    # Remove trailing/leading underscores, periods, commas, and dashes
    while text and text[-1] in ('。', '.', '，', ',', '_', '—', '-', ':', '：'):
        text = text[:-1].strip()
    while text and text[0] in ('_', '—', '-', ':', '：', '，', ','):
        text = text[1:].strip()
        
    # Replace common literal translation errors
    text = text.replace("是键", "是关键")
    text = text.replace("是核心键", "是核心")
    
    # Canonicalize proper nouns in output
    for wrong, right in _PROPER_NOUNS.items():
        text = _replace_term(text, wrong, right)
    
    return _manga_punct(text.strip())

def _detect_src_lang(text: str) -> str:
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return "auto"
    ascii_ratio = sum(1 for c in alpha if ord(c) < 128) / len(alpha)
    return "en" if ascii_ratio > 0.6 else "auto"

def _is_gibberish(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if len(t) <= 3:
        return False
        
    vowels = set("aeiouAEIOU")
    
    # 1. If the entire text contains letters but has absolutely no vowels, it's gibberish (e.g. RMMBBL, TMPYh)
    alpha_chars = [c for c in t if c.isalpha()]
    has_cjk = any('\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' for c in t)
    
    if alpha_chars and not any(c in vowels for c in alpha_chars) and not has_cjk:
        return True
        
    # 2. If it contains weird OCR structural symbols, it is gibberish
    weird_chars = {'{', '}', '[', ']', '|', '\\', '<', '>'}
    if any(c in weird_chars for c in t):
        return True
        
    # 3. Word-by-word density check
    words = t.split()
    if not words:
        return True
        
    gibberish_count = 0
    for w in words:
        # Check if the word is a pure symbol mess
        weird_chars_in_word = sum(1 for c in w if c in '{}[|]\\<>/#*')
        if weird_chars_in_word > 0.5 * len(w):
            gibberish_count += 1
            continue
            
        w_clean = "".join(c for c in w if c.isalpha())
        w_has_cjk = any('\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' for c in w)
        if len(w_clean) >= 4 and not w_has_cjk:
            if not any(c in vowels for c in w_clean):
                gibberish_count += 1
                
    if len(words) == 1:
        return gibberish_count > 0
        
    return (gibberish_count / len(words)) > 0.5

def _is_noise(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
        
    has_cjk = any('\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' for c in t)
    
    if not has_cjk and len(t) < 2:
        return True
    tl = t.lower()
    
    # Filter pure gibberish/OCR errors
    if _is_gibberish(t):
        return True
    
    # Filter internet archive / URLs
    if "archive.org" in tl or "http" in tl or "www." in tl or ".org" in tl or ".com" in tl or "details" in tl:
        return True
    if "/" in t and len(t) > 15 and t.count("/") > 2:
        return True
        
    # Filter short vowel-less strings as noise (like TMP, P, Tm, Shh, Hmmm)
    alpha = [c for c in t if c.isalpha()]
    has_cjk = any('\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' for c in t)
    
    if alpha and not any(c in "aeiouAEIOU" for c in alpha) and not has_cjk:
        # Allow exceptions like "Mr", "Dr", "Vs"
        if tl not in {"mr", "mr.", "dr", "dr.", "vs", "vs."}:
            return True
            
    # Common SFX / noise words (only if the block is very short, to avoid filtering real sentences)
    noise_keywords = {
        "whipp", "krnch", "mrnch", "krngh", "yaaawn", "shhh", "hmmm", 
        "kencu", "kench", "shash", "fwsh", "thump", "whump", "bam", "pow", 
        "zap", "smash", "crash", "boom", "bang", "clang", "clash", "click", 
        "snap", "crackle", "pop", "whoosh", "swish", "splat", "squish", 
        "gasp", "sigh", "pant", "gulp", "smack", "kiss", "muah", "grrr", 
        "roar", "bark", "meow", "purr", "chirp", "tweet", "buzz", "hiss",
        "drip", "drop", "plop", "splash", "fizz", "sizzle", "ding", "dong",
        "ring", "beep", "honk", "toot", "vroom", "screech", "thud", "thwack",
        "wham", "biff", "sock", "kapow", "zonk", "boing", "boink", "sproing",
        "rustle", "creak", "squeak", "groan", "moan", "whimper", "sob",
        "sniff", "snort", "sneeze", "cough", "hiccup", "burp", "fart", "tmp"
    }
    words_lower = tl.split()
    if len(words_lower) <= 3:
        # Check if any word exactly matches or contains a noise keyword prominently
        for w in words_lower:
            w_alpha = "".join(c for c in w if c.isalpha())
            if w_alpha in noise_keywords:
                return True
        
    if not has_cjk:
        # Nearly no alphabetic content (only apply to Latin/English text)
        if not alpha:
            return True
        if len(alpha) / len(t) < 0.35:
            return True
    else:
        # For CJK, it is noise ONLY if there are literally 0 CJK characters
        if not any('\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' for c in t):
            return True
    return False

def _google_translate_one(text: str, src: str, tgt: str = "zh-CN") -> str:
    """Call the free Google Translate API endpoint (timed into the batch breakdown)."""
    t0 = time.time()
    try:
        out = _google_translate_one_raw(text, src, tgt)
        _stats_add("google", time.time() - t0)
        return out
    except Exception:
        _stats_add("google", time.time() - t0, ok=False)
        raise


def _google_translate_one_raw(text: str, src: str, tgt: str = "zh-CN") -> str:
    url = "https://translate.googleapis.com/translate_a/single"
    params = {"client": "gtx", "sl": src, "tl": tgt, "dt": "t", "q": text}
    try:
        r = httpx.get(url, params=params, timeout=10.0)
        r.raise_for_status()
        data = r.json()
        parts = [seg[0] for seg in data[0] if seg[0]]
        return "".join(parts).strip()
    except Exception as e:
        raise RuntimeError(f"Google API error: {e}") from e


def _should_polish(raw_text: str, draft: str) -> bool:
    """
    多维启发式评分：返回 True 表示值得调用 Qwen 润色。
    目标：过滤掉约 50% 的短句/简单句，减少不必要的 LLM 推理。
    """
    words = raw_text.split()

    # 1. 纯数字或代码段：不需要自然语言润色
    if raw_text.replace(" ", "").isdigit():
        return False

    # 3. 大量全大写专有名词（>50%）：如 KAFNA, TEPEL 等角色/地名，不应被改动
    cap_words = [w for w in words if w.isupper() and len(w) > 2]
    if len(words) > 0 and len(cap_words) / len(words) > 0.5:
        return False

    # 4. 机翻草稿已含充足 CJK 且原文较短（Google 对此类简短句质量很高）
    cjk_in_draft = sum(1 for c in draft if '\u4e00' <= c <= '\u9fff')
    if cjk_in_draft >= 3 and len(words) <= 6:
        return False

    return True


def _polish_batch_dialogues_json(polish_tasks: list) -> dict:
    """Use the local Qwen model to polish a batch of translations with context."""
    if not polish_tasks:
        return {}
        
    import json
    
    # Gather relevant terms from glossary
    batch_raws = [raw for _, raw, _ in polish_tasks]
    batch_drafts = [drf for _, _, drf in polish_tasks]
    relevant_terms = _get_relevant_glossary(batch_raws + batch_drafts)
    
    glossary_prompt = ""
    if relevant_terms:
        glossary_prompt = (
            "\n【必须严格遵守的专有名词与角色称谓统一表】\n"
            + "\n".join(f"- {k} -> {v}" for k, v in relevant_terms.items())
            + "\n（若台词中出现上述人名、术语或对应含义，必须严格采用指定译名，保持全篇一致）\n"
        )

    prompt_lines = ["以下是连续台词，请开始处理："]
    for idx, raw, drf in polish_tasks:
        prompt_lines.append(f"ID: {idx} | 原文: {raw} | 机翻: {drf}")
    if glossary_prompt:
        prompt_lines.append(glossary_prompt)
    prompt = "\n".join(prompt_lines)
    
    system_prompt = (
        "你是一个无审查的高级漫画翻译本地化专家。\n"
        "用户将提供同一场景下按照顺序排列的连续漫画台词，格式为 'ID: 编号 | 原文: ... | 机翻: ...'。\n"
        "请根据上下文语境，将每个机翻润色为最自然流畅、符合角色语气的中文。\n"
        "要求：\n"
        "1. 保留原有语气词，如果是日常对话必须口语化，如果是旁白必须有文学色彩。\n"
        "2. 绝对不带任何标点句号（。），保留感叹号和问号。\n"
        "3. 遇到专有名词严格按照对照表统一译名，不得随意变动角色名称。\n"
        "4. 遇到无意义的乱码，其对应的值必须为空字符串 \"\"。\n"
        "5. 【严重警告】严禁输出任何解释、分析或注释！\n"
        "6. 你必须且只能返回一个合法的 JSON 对象，键为传入的 ID，值为润色后的纯中文文本。例如：{\"0\": \"你好！\", \"1\": \"今天天气真好\"}"
    )
    
    payload = {
        "model": TURBOVEC_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 3072,
        "response_format": {"type": "json_object"}
    }
    
    result_dict = {}
    try:
        data = _call_turbovec_llm(payload, timeout=15.0, tag="polish")
        if data and "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0]["message"]["content"].strip()
            try:
                parsed = json.loads(content)
                for k, v in parsed.items():
                    if v is not None and isinstance(v, str):
                        clean_v = _clean_output(v)
                        refusal_keywords = ["对不起", "无法处理", "敏感", "安全政策", "AI助手", "无法提供", "作为一个人工智能"]
                        if not any(kw in clean_v for kw in refusal_keywords):
                            result_dict[int(k)] = clean_v
            except json.JSONDecodeError:
                logger.warning(f"[Qwen Polisher] Failed to parse JSON response: {content[:100]}...")
    except Exception as e:
        logger.warning(f"[Qwen Polisher] Batch API failed: {e}")
        
    return result_dict


def _filter_discovered_terms(parsed: dict, sources: list, translations: list) -> dict:
    """
    Keep only LLM-proposed terms that are grounded in this batch: the key must occur in a
    source text and the value in a translation, and the pair must pass _is_valid_term.
    """
    src_all = "\n".join(sources)
    trans_all = "\n".join(translations)
    kept = {}
    for k, v in (parsed or {}).items():
        k_clean, v_clean = str(k).strip(), str(v).strip()
        if not _is_valid_term(k_clean, v_clean):
            continue
        if k_clean in _PROPER_NOUNS or k_clean in _AUTO_TERMS:
            continue
        if not _term_in_text(k_clean, src_all) or v_clean not in trans_all:
            continue
        kept[k_clean] = v_clean
    return kept


def _save_auto_terms():
    """Atomic write of the auto-discovered terms file (caller holds _TERMS_LOCK)."""
    tmp = _AUTO_TERMS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_AUTO_TERMS, f, ensure_ascii=False, indent=4)
    os.replace(tmp, _AUTO_TERMS_PATH)


def _extract_proper_nouns_from_batch(blocks: list, results: list):
    """
    用 LLM 分析刚才这批翻译，提取可能遗漏的专有名词，写入 proper_nouns_auto.json。
    这些词条未经人工确认，只作为后续 prompt 的参考，不会强制替换原文/译文（BUG.md B5）。
    设置环境变量 AUTO_PROPER_NOUNS=0 可关闭（省去每批一次 LLM 调用）。
    """
    if not AUTO_PROPER_NOUNS_ENABLED:
        return
    pairs, sources, translations = [], [], []
    for i, block in enumerate(blocks):
        if block.get("is_sfx"): continue
        orig = block.get("cleaned_text", "").strip()
        trans = results[i].strip()
        if orig and trans and len(orig) > 4:
            pairs.append(f"原文: {orig} | 译文: {trans}")
            sources.append(orig)
            translations.append(trans)
            
    if not pairs:
        return
        
    prompt = "以下是一批漫画翻译对话。请提取其中明显的【特有专有名词】（如角色名、地名、特有招式等），特别是全大写的英文/罗马音单词或特殊称谓。如果没有发现专有名词，请返回空字典。严格输出纯 JSON，格式如: {\"KAFNA\": \"卡夫娜\", \"TEPEL\": \"泰佩尔\"}。\n\n" + "\n".join(pairs)
    
    payload = {
        "model": TURBOVEC_MODEL,
        "messages": [
            {"role": "system", "content": "你是一个专有名词提取器。只返回JSON，禁止任何其他文字。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 512,
        "response_format": {"type": "json_object"}
    }
    try:
        data = _call_turbovec_llm(payload, timeout=8.0, tag="proper_nouns")
        if data and "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0]["message"]["content"].strip()
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                return
            if not isinstance(parsed, dict):
                return
            with _TERMS_LOCK:
                kept = _filter_discovered_terms(parsed, sources, translations)
                if kept:
                    _AUTO_TERMS.update(kept)
                    try:
                        _save_auto_terms()
                        logger.info(f"[Dynamo] +{len(kept)} auto terms (hint-only): {kept}")
                    except Exception as e:
                        logger.warning(f"[Dynamo] Failed to save auto terms: {e}")
    except Exception as e:
        logger.warning(f"[Dynamo] Proper noun extraction failed: {e}")


# Chinese scanlation (汉化组) lettering style for Japanese manga (user request 2026-09-28).
MANGA_SYSTEM_PROMPT = (
    "你是资深日漫汉化组的翻译兼嵌字编辑，把日文漫画台词译成中文汉化版里的台词。\n"
    "规则：\n"
    "1. 按同一场景的连续对话理解语境，译成符合人物身份、性格、情绪的口语化中文；旁白、说明文字可以书面一些。\n"
    "2. 句子短而有力，和原文长度差不多，不要扩写、不要解释，能省的主语和连词就省掉。\n"
    "3. 保留语气：用「啊、呢、吧、嘛、哦、呀、啦、诶、哈」等语气词表现原文的「ね、よ、な、ぞ、わ、かな」。\n"
    "4. 标点用漫画习惯：不用句号「。」；省略、停顿用「……」；拖长音用「～」；惊讶用「！？」；强调可以用「！！」；句中停顿用空格或「，」。\n"
    "5. 称呼本土化：さん→（按语境）先生/小姐/桑或省略，ちゃん→酱/小～，君→君/小～，様→大人/阁下；不得留罗马音或英文。\n"
    "6. 拟声拟态词（ドキッ、ゴゴゴ 等）译成中文象声词（扑通、轰隆隆 等）。\n"
    "7. 专有名词严格按对照表统一译名。\n"
    "8. 只返回 JSON 对象，不要任何解释。\n"
    "示例：\n"
    "原文: 以上が所蔵から排架までの本の受け入れの流れだ → 以上就是从馆藏到上架的收书流程\n"
    "原文: え…嘘でしょ!? → 诶……骗人的吧！？\n"
    "原文: 今日は一番を目指す!! → 今天我要拿第一！！\n"
    "原文: そんなことより早く行こうよ → 别管那些了，快走吧"
)

_ELLIPSIS_RE = re.compile(r"(?:\.{2,}|。{2,}|…+|・{3,}|‥+)")


def _manga_punct(text: str) -> str:
    """Normalise punctuation to Chinese manga lettering conventions."""
    if not text:
        return text
    t = _ELLIPSIS_RE.sub("……", text)
    t = t.replace("!?", "！？").replace("?!", "！？").replace("？！", "！？")
    t = t.replace("!", "！").replace("?", "？").replace("~", "～").replace("〜", "～")
    t = re.sub(r"(……){2,}", "……", t)
    t = re.sub(r"[，、,]+$", "", t)                      # no dangling comma at the end
    t = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", " ", t)  # single space between CJK
    return t.strip()


MIN_CONTEXT_CHUNK = 12
UNTRANSLATED_ENGINE = "Untranslated (all engines failed)"
_JSON_SCHEMA_OK = True       # flips to False if the server rejects json_schema (HTTP 400)


def _batch_max_tokens(n_lines: int) -> int:
    """
    Output budget for an n-line JSON batch: <=80 CJK chars (~60 tokens) + key/quotes per line.
    Kept tight on purpose: with json_object mode a model can emit endless whitespace until
    max_tokens, and on a single-slot server that runaway blocks every later request.
    """
    return int(min(4096, 64 + 72 * n_lines))


def _disable_json_schema():
    global _JSON_SCHEMA_OK
    _JSON_SCHEMA_OK = False


def _batch_response_format(ids) -> dict:
    """
    Grammar-bounded JSON: exactly these keys, string values, no extra properties.
    llama.cpp's schema grammar also bounds whitespace between tokens.
    """
    if not _JSON_SCHEMA_OK:
        return {"type": "json_object"}
    keys = [str(i) for i in ids]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "translations",
            "strict": True,
            "schema": {
                "type": "object",
                # No maxLength: a {0,80} repetition per key blows up the grammar and made
                # 15-30 key batches crawl past their timeout (serial mode). Length is bounded
                # by max_tokens + streaming timeout instead.
                "properties": {k: {"type": "string"} for k in keys},
                "required": keys,
                "additionalProperties": False,
            },
        },
    }


def _batch_timeout(n_lines: int) -> float:
    """Generation at ~35-50 tok/s needs longer timeouts for bigger batches."""
    return float(max(18.0, 1.2 * n_lines + 10.0))


class HighPerformanceTranslationEngine:
    """
    Primary: Google Translate API (30 concurrent workers).
    Fallback: OPUS-MT local offline model.
    Final: Qwen LLM (strictly constrained, no chat).
    Same public interface as all previous versions.
    """

    def __init__(self, use_gpu=True):
        self.use_gpu = False
        self.device = "cpu"
        self.model_name = "Google Translate API + OPUS-MT Fallback"
        self._executor = ThreadPoolExecutor(max_workers=30)
        logger.info("[Engine v7] Google API (primary) + OPUS-MT (fallback) ready.")
        logger.info("[Google API] Translation engine initialized - 30 concurrent workers, no model download needed.")

    def translate_batch(self, blocks, source_lang="English", target_lang="Simplified Chinese",
                        context_chunk_size: int = 12):
        """
        context_chunk_size: how many Japanese lines go into one LLM call (more = more dialogue
        context). If a large chunk fails (e.g. exceeds the server context), it is split in half
        and retried down to MIN_CONTEXT_CHUNK before falling back to single-line / Google.
        """
        if not blocks:
            return [], self._empty_metrics()

        import time
        t0 = time.time()
        _stats_begin()
        results = [""] * len(blocks)

        # ── Step 1: Pre-clean all blocks ─────────────────────────────────────
        to_translate = []
        for i, block in enumerate(blocks):
            orig_text = block.get("text", "").strip()
            
            # Exact match in proper nouns JSON -> direct assignment
            if orig_text and orig_text in _PROPER_NOUNS:
                results[i] = _PROPER_NOUNS[orig_text]
                block["cleaned_text"] = orig_text
                block["google_trans"] = results[i]
                block["is_sfx"] = False
                block["translation_engine"] = "Glossary (专有名词库)"
                continue

            raw = _clean_input(orig_text)
            block["cleaned_text"] = raw
            block["google_trans"] = ""
            block["is_sfx"] = False

            if not raw:
                continue
            
            has_cjk = any('\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' for c in raw)
            if len(raw) < 2 and not has_cjk:
                continue
                
            if _is_noise(raw):
                block["is_sfx"] = True
                block["translation_engine"] = "SFX Noise Filter (音效跳过)"
                continue
            if raw.isdigit() and len(raw) <= 4:
                results[i] = raw
                block["google_trans"] = raw
                block["translation_engine"] = "Numeric Passthrough (纯数字)"
                continue

            src = _detect_src_lang(raw)
            to_translate.append((i, raw, src))

        # ── Step 2-4: Translation Pipeline ──
        if source_lang == "Japanese":
            # ── Japanese-specific routing: Local Qwen Context-Batch Cascade ──
            qwen_needed = []
            for i, block in enumerate(blocks):
                if block.get("is_sfx"):
                    continue
                if results[i]:
                    continue
                raw = block.get("cleaned_text", "").strip()
                if raw and any(c.isalpha() or '\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff' for c in raw):
                    qwen_needed.append(i)

            if qwen_needed:
                ensure_turbovec_llm_ready(auto_launch=True, max_wait_seconds=20)
                logger.info(f"[Qwen Japanese Translation] Translating {len(qwen_needed)} blocks in context batch...")
                chunk_size = max(1, int(context_chunk_size))
                pending_chunks = [qwen_needed[c:c + chunk_size] for c in range(0, len(qwen_needed), chunk_size)]
                while pending_chunks:
                    chunk_indices = pending_chunks.pop(0)
                    batch_texts = [blocks[idx].get("cleaned_text", "").strip() for idx in chunk_indices]
                    
                    # Match relevant terminology for this batch
                    relevant_glossary = _get_relevant_glossary(batch_texts)
                    glossary_prompt = ""
                    if relevant_glossary:
                        glossary_prompt = (
                            "\n【必须严格遵守的专有名词与角色称谓统一表】:\n"
                            + "\n".join(f"- {k} -> {v}" for k, v in relevant_glossary.items())
                            + "\n"
                        )
                    
                    prompt_lines = ["以下是同一场景下的连续日语漫画台词，请根据连续对话上下文将其翻译为最自然流畅、符合角色语气性格的中文台词："]
                    for idx in chunk_indices:
                        raw_t = blocks[idx].get("cleaned_text", "").strip()
                        prompt_lines.append(f"ID: {idx} | 原文: {raw_t}")
                    if glossary_prompt:
                        prompt_lines.append(glossary_prompt)
                    prompt_lines.append("\n请严格返回一个合法的 JSON 对象，键为传入的 ID，值为对应的纯中文翻译文本。例如：{\"0\": \"你好！\", \"1\": \"今天天气真好\"}")
                    user_prompt = "\n".join(prompt_lines)
                    
                    system_prompt = MANGA_SYSTEM_PROMPT
                    
                    payload = {
                        "model": TURBOVEC_MODEL,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0.2,
                        "max_tokens": _batch_max_tokens(len(chunk_indices)),
                        "response_format": _batch_response_format(chunk_indices)
                    }
                    
                    batch_llm_success = False
                    for attempt in range(1, 3):
                        try:
                            data = _call_turbovec_llm(payload, timeout=_batch_timeout(len(chunk_indices)), tag="batch")
                            if (data is None and getattr(_TLS, "last_status", None) == 400
                                    and payload["response_format"].get("type") == "json_schema"):
                                _disable_json_schema()
                                logger.warning("[Qwen Batch] server rejected json_schema; falling back to json_object")
                                payload["response_format"] = {"type": "json_object"}
                                data = _call_turbovec_llm(payload, timeout=_batch_timeout(len(chunk_indices)), tag="batch")
                            if data and "choices" in data and len(data["choices"]) > 0:
                                batch_llm_success = True
                                reply = data["choices"][0]["message"]["content"].strip()
                                parsed = json.loads(reply)
                                for k, v in parsed.items():
                                    try:
                                        k_int = int(k)
                                        if k_int in chunk_indices and v and isinstance(v, str):
                                            clean_v = _clean_output(v)
                                            if any('\u4e00' <= c <= '\u9fff' for c in clean_v):
                                                results[k_int] = clean_v
                                                blocks[k_int]["translation_engine"] = "Turbovec Qwen 3.5 4B (Context Batch)"
                                                logger.info(f"[Qwen Batch OK] ID {k_int} → '{clean_v}'")
                                    except (ValueError, TypeError):
                                        pass
                                break
                        except Exception as ex:
                            logger.warning(f"[Qwen Batch Translation] Attempt {attempt} failed: {ex}")
                            time.sleep(0.5)
                            
                    # Large chunk failed as a whole (context overflow / timeout): split and retry
                    if not batch_llm_success and len(chunk_indices) > MIN_CONTEXT_CHUNK:
                        half = (len(chunk_indices) + 1) // 2
                        logger.warning(f"[Qwen Batch] {len(chunk_indices)}-line chunk failed, retrying as {half} + {len(chunk_indices) - half}")
                        pending_chunks[:0] = [chunk_indices[:half], chunk_indices[half:]]
                        continue

                    # Fallback for any untranslated blocks in this batch
                    for idx in chunk_indices:
                        if not results[idx]:
                            raw_text = blocks[idx].get("cleaned_text", "").strip()
                            if not raw_text:
                                continue
                            # 1. Single-sentence LLM fallback if LLM batch succeeded
                            if batch_llm_success:
                                single_prompt = f"原文：{raw_text}\n请直接给出最完美的本地化中文译文："
                                single_payload = {
                                    "model": TURBOVEC_MODEL,
                                    "messages": [
                                        {"role": "system", "content": system_prompt},
                                        {"role": "user", "content": single_prompt}
                                    ],
                                    "temperature": 0.2,
                                    "max_tokens": 120,
                                }
                                try:
                                    sdata = _call_turbovec_llm(single_payload, timeout=8.0, tag="single_fallback")
                                    if sdata and "choices" in sdata and len(sdata["choices"]) > 0:
                                        sc = sdata["choices"][0]["message"]["content"].strip()
                                        if any('\u4e00' <= c <= '\u9fff' for c in sc):
                                            results[idx] = _clean_output(sc)
                                            blocks[idx]["translation_engine"] = "Turbovec Qwen 3.5 4B (Single Fallback)"
                                except Exception as sex:
                                    logger.warning(f"[Qwen Single Fallback Failed] '{raw_text[:30]}': {sex}")

                            # 2. Safety fallback: Google Translate API (guarantees text is translated even if LLM is down)
                            if not results[idx]:
                                try:
                                    g_res = _google_translate_one(raw_text, src="ja", tgt="zh-CN")
                                    if g_res and any('\u4e00' <= c <= '\u9fff' for c in g_res):
                                        results[idx] = _clean_output(g_res)
                                        blocks[idx]["translation_engine"] = "Google Translate API (Fallback)"
                                        logger.info(f"[Google Fallback OK] ID {idx} → '{results[idx]}'")
                                except Exception as g_err:
                                    logger.warning(f"[Google Fallback Failed] ID {idx}: {g_err}")

                            # 3. All engines failed: leave the block UNtranslated (empty). The renderer then
                            #    keeps the original bubble untouched instead of erasing it and redrawing the
                            #    same Japanese in a Chinese font, and nothing wrong gets cached (BUG.md B8).
                            if not results[idx] and raw_text:
                                blocks[idx]["translation_engine"] = UNTRANSLATED_ENGINE

            # Note: For Japanese, Qwen directly outputs localized, context-aware manga dialogues with glossary adherence.
            # Skipping the redundant 2nd polishing pass saves 5-15s per page and prevents dropping lines.

        else:
            # ── English/Other Language pipeline: Google API + LLM Preprocessor + OPUS-MT Fallback ──
            if to_translate:
                future_to_idx = {}
                for idx, text, src in to_translate:
                    fut = self._executor.submit(_google_translate_one, text, src, "zh-CN")
                    future_to_idx[fut] = idx
                for fut in as_completed(future_to_idx):
                    idx = future_to_idx[fut]
                    try:
                        trans_res = fut.result()
                        if trans_res:
                            blocks[idx]["google_trans"] = trans_res
                    except Exception as e:
                        logger.warning(f"Google translate failed for block {idx}: {e}")

            # Removed LLMTextPreprocessor because a 1.5B model is too unstable for complex JSON OCR correction
            # and causes hallucinations like '创作者：创作者'. We rely directly on Google Translate which the user noted was perfect.

            for i, block in enumerate(blocks):
                trans = block.get("translated_text", "").strip()
                if trans:
                    results[i] = trans
                elif block.get("is_sfx"):
                    results[i] = ""
                else:
                    g_trans = block.get("google_trans", "").strip()
                    if g_trans and any('\u4e00' <= c <= '\u9fff' for c in g_trans):
                        results[i] = _clean_output(g_trans)
                        block["translated_text"] = results[i]

            opus_needed = []
            for i, block in enumerate(blocks):
                if block.get("is_sfx"):
                    continue
                if not results[i]:
                    opus_needed.append(i)

            if opus_needed:
                try:
                    from core.local_translator import get_local_translator
                    lt = get_local_translator()
                    texts_to_translate = [blocks[idx]["cleaned_text"] for idx in opus_needed]
                    opus_results = lt.translate_batch(texts_to_translate)
                    for idx, trans_res in zip(opus_needed, opus_results):
                        if trans_res:
                            results[idx] = trans_res
                            blocks[idx]["translated_text"] = trans_res
                except Exception as e:
                    logger.warning(f"OPUS-MT translation failed: {e}")

            # ── Step 4.5: Qwen polishing for conversational quality (English) ──
            polish_tasks = []
            for i, block in enumerate(blocks):
                if block.get("is_sfx"):
                    continue
                draft = results[i]
                if not draft:
                    continue
                raw_text = block.get("cleaned_text", "").strip()
                if _should_polish(raw_text, draft):
                    polish_tasks.append((i, raw_text, draft))

            if polish_tasks:
                logger.info(f"[Qwen Polisher] Context-Aware Polishing {len(polish_tasks)} English blocks in a single JSON batch...")
                polished_results = _polish_batch_dialogues_json(polish_tasks)
                for idx, _, draft in polish_tasks:
                    if idx in polished_results and polished_results[idx]:
                        results[idx] = polished_results[idx]
                    else:
                        results[idx] = draft

        # ── Step 5: Write results back to blocks ──────────────────────────────
        for i, block in enumerate(blocks):
            block["translated_text"] = results[i]
            if not block.get("translation_engine"):
                if results[i]:
                    block["translation_engine"] = "Turbovec Qwen 3.5 4B" if source_lang == "Japanese" else "Google Translate API"
                elif block.get("is_sfx"):
                    block["translation_engine"] = "SFX Noise Filter (音效跳过)"
                else:
                    block["translation_engine"] = "Passthrough / Untranslated"

        # Model breakdown calculation
        engine_counts = {}
        for b in blocks:
            eng = b.get("translation_engine", "Unknown")
            engine_counts[eng] = engine_counts.get(eng, 0) + 1

        # ── Step 6: Dynamically Extract Proper Nouns ─────────────────────────
        _extract_proper_nouns_from_batch(blocks, results)
        breakdown = _stats_end()

        elapsed = time.time() - t0
        total_chars = sum(len(r) for r in results)
        metrics = {
            "inference_mode": "Google API + LLM Preprocessor + OPUS-MT / Qwen Cascade",
            "tokens_per_sec": round(total_chars / max(0.01, elapsed), 2),
            "acceptance_rate": 100.0,
            "latency_ms": round(elapsed * 1000, 2),
            "cuda_graphs_active": False,
            "tokens_generated": total_chars,
            "model_breakdown": engine_counts,
            "time_breakdown": breakdown
        }
        logger.info(
            f"[✓] {len(blocks)} blocks translated in {elapsed:.2f}s "
            f"({metrics['tokens_per_sec']:.0f} chars/s) | Models: {engine_counts} | Time: {breakdown}"
        )
        return results, metrics

    def _empty_metrics(self):
        return {
            "inference_mode": "N/A",
            "tokens_per_sec": 0.0,
            "acceptance_rate": 0.0,
            "latency_ms": 0.0,
            "cuda_graphs_active": False,
            "tokens_generated": 0
        }

if __name__ == "__main__":
    engine = HighPerformanceTranslationEngine()
    test_blocks = [
        {"id": 0, "text": "Wasn't I clear enough? You are a member of the Kafna!"},
        {"id": 1, "text": "I couldn't help it! Seeing how true devotees react to outsiders."},
        {"id": 2, "text": "WHIPP"},
        {"id": 3, "text": "Sir Blakk, we found something."},
        {"id": 4, "text": "Even if they are on the shelves; we still need to inspect them."},
    ]
    t0 = time.time()
    results, metrics = engine.translate_batch(test_blocks)
    for b, r in zip(test_blocks, results):
        print(f"  '{b['text'][:50]}' → '{r}'")
    print(f"\n[✓] {len(test_blocks)} blocks in {time.time()-t0:.2f}s")
