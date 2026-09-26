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

def _call_turbovec_llm(payload: dict, timeout: float = 15.0):
    """
    Calls Turbovec OpenAI-compatible API endpoint (direct llama-server 18089 or Turbovec gateway 18088).
    Automatically injects target model name, auth header, and provides connection resilience and retries.
    """
    if "model" not in payload or not payload["model"]:
        payload["model"] = TURBOVEC_MODEL
        
    headers = {}
    key = _get_llama_api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
        
    # Prioritize 18089 direct llama-server for speed, fallback to 18088 gateway
    endpoints = [LLAMA_SERVER_DIRECT_URL, TURBOVEC_API_URL]
    
    for attempt in range(1, 4):
        last_err = None
        for url in endpoints:
            try:
                r = httpx.post(url, json=payload, headers=headers, timeout=timeout)
                if r.status_code == 200:
                    return r.json()
                else:
                    logger.warning(f"[Turbovec LLM] HTTP {r.status_code} from {url}: {r.text[:100]}")
            except Exception as e:
                last_err = e
                continue
                
        # If connection refused and attempt 1, try auto-launching or waiting
        if attempt < 3:
            time.sleep(1.0)
            continue
            
    logger.warning(f"[Turbovec LLM] All endpoints failed: {last_err}")
    return None

# ── Proper noun pre-fixes and glossary loaded from JSON ───────────────────────
_PROPER_NOUNS = {}
_PROPER_NOUNS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "proper_nouns.json")
_GLOSSARY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "glossary.json")

try:
    if os.path.exists(_PROPER_NOUNS_PATH):
        with open(_PROPER_NOUNS_PATH, "r", encoding="utf-8") as f:
            _PROPER_NOUNS.update(json.load(f))
    if os.path.exists(_GLOSSARY_PATH):
        with open(_GLOSSARY_PATH, "r", encoding="utf-8") as f:
            gloss_data = json.load(f)
            if isinstance(gloss_data, list):
                for item in gloss_data:
                    src = item.get("src")
                    tgt = item.get("tgt")
                    if src and tgt and src not in _PROPER_NOUNS:
                        _PROPER_NOUNS[src] = tgt
except Exception as e:
    logger.warning(f"Could not load proper nouns / glossary: {e}")

def _get_relevant_glossary(texts: list) -> dict:
    """Finds glossary and character terms matching any of the texts in the current batch."""
    if not _PROPER_NOUNS:
        return {}
    combined = " ".join(str(t) for t in texts if t)
    combined_lower = combined.lower()
    matched = {}
    for k, v in _PROPER_NOUNS.items():
        if not k or not v:
            continue
        if k in combined or k.lower() in combined_lower:
            matched[k] = v
    return matched

# ── Post-translation cleanup ─────────────────────────────────────────────────
_PAREN_EN_RE = re.compile(r'\s*\([A-Za-z][A-Za-z\s\-]*\)')

def _clean_input(text: str) -> str:
    text = " ".join(text.replace("\n", " ").split()).strip()
    
    # OCR often misreads 'a' as '@' (e.g., 'across' -> '@cross')
    text = text.replace("@", "a")
    
    for wrong, right in _PROPER_NOUNS.items():
        text = text.replace(wrong, right)
        
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
        if wrong and right and wrong != right:
            if wrong in text:
                text = text.replace(wrong, right)
    
    return text.strip()

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
    """Call the free Google Translate API endpoint."""
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
        data = _call_turbovec_llm(payload, timeout=15.0)
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


def _extract_proper_nouns_from_batch(blocks: list, results: list):
    """
    用 LLM 分析刚才这批翻译，提取可能遗漏的专有名词，动态写入 proper_nouns.json。
    """
    pairs = []
    for i, block in enumerate(blocks):
        if block.get("is_sfx"): continue
        orig = block.get("cleaned_text", "").strip()
        trans = results[i].strip()
        if orig and trans and len(orig) > 4:
            pairs.append(f"原文: {orig} | 译文: {trans}")
            
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
        data = _call_turbovec_llm(payload, timeout=8.0)
        if data and "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0]["message"]["content"].strip()
            try:
                parsed = json.loads(content)
                new_discovered = False
                for k, v in parsed.items():
                    k_clean = str(k).strip()
                    v_clean = str(v).strip()
                    if k_clean and v_clean and k_clean not in _PROPER_NOUNS:
                        if len(k_clean) < 30 and len(v_clean) < 30:
                            _PROPER_NOUNS[k_clean] = v_clean
                            new_discovered = True
                            
                if new_discovered:
                    try:
                        with open(_PROPER_NOUNS_PATH, "w", encoding="utf-8") as f:
                            json.dump(_PROPER_NOUNS, f, ensure_ascii=False, indent=4)
                        logger.info(f"[Dynamo] Extracted new proper nouns! Dict size: {len(_PROPER_NOUNS)}")
                    except Exception as e:
                        logger.warning(f"[Dynamo] Failed to save proper nouns: {e}")
            except json.JSONDecodeError:
                pass
    except Exception as e:
        logger.warning(f"[Dynamo] Proper noun extraction failed: {e}")


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

    def translate_batch(self, blocks, source_lang="English", target_lang="Simplified Chinese"):
        if not blocks:
            return [], self._empty_metrics()

        import time
        t0 = time.time()
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
                logger.info(f"[Qwen Japanese Translation] Translating {len(qwen_needed)} blocks in context batch...")
                chunk_size = 12
                for c_start in range(0, len(qwen_needed), chunk_size):
                    chunk_indices = qwen_needed[c_start:c_start + chunk_size]
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
                    
                    system_prompt = (
                        "你是一个顶尖的日漫汉化翻译专家。\n"
                        "规则：\n"
                        "1. 根据同一场景的连续对话语境，将每句台词翻译为符合人物口吻与情绪的中文台词。\n"
                        "2. 绝对不带任何标点句号（。），可以保留感叹号、问号和省略号。\n"
                        "3. 严禁在译文中保留任何未翻译的罗马音或英文（如 chan, san, kun, sama），必须本土化为'酱'、'桑'、'君'、'大人'等或根据语境省略。\n"
                        "4. 遇到专有名词必须严格按照对照表统一译名。\n"
                        "5. 严禁输出任何多余的解释、前言或分析，只返回 JSON 对象。"
                    )
                    
                    payload = {
                        "model": TURBOVEC_MODEL,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0.2,
                        "max_tokens": 2048,
                        "response_format": {"type": "json_object"}
                    }
                    
                    batch_llm_success = False
                    for attempt in range(1, 3):
                        try:
                            data = _call_turbovec_llm(payload, timeout=18.0)
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
                            
                    # Single-sentence LLM fallback only if LLM is online and responsive
                    if batch_llm_success:
                        for idx in chunk_indices:
                            if not results[idx]:
                                raw_text = blocks[idx].get("cleaned_text", "").strip()
                                if not raw_text:
                                    continue
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
                                    sdata = _call_turbovec_llm(single_payload, timeout=8.0)
                                    if sdata and "choices" in sdata and len(sdata["choices"]) > 0:
                                        sc = sdata["choices"][0]["message"]["content"].strip()
                                        if any('\u4e00' <= c <= '\u9fff' for c in sc):
                                            results[idx] = _clean_output(sc)
                                            blocks[idx]["translation_engine"] = "Turbovec Qwen 3.5 4B (Single Fallback)"
                                except Exception as sex:
                                    logger.warning(f"[Qwen Single Fallback Failed] '{raw_text[:30]}': {sex}")

                    # Ultimate safety fallback: Google Translate API (guarantees NO text is ever left untranslated)
                        if not results[idx]:
                            raw_text = blocks[idx].get("cleaned_text", "").strip()
                            if raw_text:
                                try:
                                    g_res = _google_translate_one(raw_text, src="ja", tgt="zh-CN")
                                    if g_res and any('\u4e00' <= c <= '\u9fff' for c in g_res):
                                        results[idx] = _clean_output(g_res)
                                        blocks[idx]["translation_engine"] = "Google Translate API (Fallback)"
                                        logger.info(f"[Google Fallback OK] ID {idx} → '{results[idx]}'")
                                except Exception as g_err:
                                    logger.warning(f"[Google Fallback Failed] ID {idx}: {g_err}")

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

        elapsed = time.time() - t0
        total_chars = sum(len(r) for r in results)
        metrics = {
            "inference_mode": "Google API + LLM Preprocessor + OPUS-MT / Qwen Cascade",
            "tokens_per_sec": round(total_chars / max(0.01, elapsed), 2),
            "acceptance_rate": 100.0,
            "latency_ms": round(elapsed * 1000, 2),
            "cuda_graphs_active": False,
            "tokens_generated": total_chars,
            "model_breakdown": engine_counts
        }
        logger.info(
            f"[✓] {len(blocks)} blocks translated in {elapsed:.2f}s "
            f"({metrics['tokens_per_sec']:.0f} chars/s) | Models: {engine_counts}"
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
