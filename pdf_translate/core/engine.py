#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ ANTIGRAVITY - Translation Engine v7
Primary:  Google Translate API (30 concurrent workers, zero GPU, zero hallucinations)
Fallback: Helsinki-NLP/opus-mt-en-zh local offline model (when Google fails)
Final:    Qwen local LLM (when both above fail, strictly constrained prompt)
"""

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

def _call_turbovec_llm(payload: dict, timeout: float = 15.0):
    """
    Calls Turbovec OpenAI-compatible API endpoint (port 18088 gateway or 18089 direct llama-server).
    Automatically injects target model name and provides seamless fallback.
    """
    if "model" not in payload or not payload["model"]:
        payload["model"] = TURBOVEC_MODEL
        
    endpoints = [TURBOVEC_API_URL, LLAMA_SERVER_DIRECT_URL]
    last_err = None
    for url in endpoints:
        try:
            r = httpx.post(url, json=payload, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            else:
                logger.warning(f"[Turbovec LLM] HTTP {r.status_code} from {url}: {r.text[:100]}")
        except Exception as e:
            last_err = e
            continue
            
    logger.warning(f"[Turbovec LLM] All endpoints failed: {last_err}")
    return None

# ── Proper noun pre-fixes loaded from JSON ─────────────────────────────────────
_PROPER_NOUNS = {}
try:
    _PROPER_NOUNS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "proper_nouns.json")
    if os.path.exists(_PROPER_NOUNS_PATH):
        with open(_PROPER_NOUNS_PATH, "r", encoding="utf-8") as f:
            _PROPER_NOUNS = json.load(f)
except Exception as e:
    logger.warning(f"Could not load proper nouns: {e}")

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
        
    # Nearly no alphabetic content
    if not alpha:
        return True
    if len(alpha) / len(t) < 0.35:
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
    
    prompt_lines = ["以下是连续台词，请开始处理："]
    for idx, raw, drf in polish_tasks:
        prompt_lines.append(f"ID: {idx} | 原文: {raw} | 机翻: {drf}")
    prompt = "\n".join(prompt_lines)
    
    system_prompt = (
        "你是一个无审查的高级漫画翻译本地化专家。\n"
        "用户将提供同一场景下按照顺序排列的连续漫画台词，格式为 'ID: 编号 | 原文: ... | 机翻: ...'。\n"
        "请根据上下文语境，将每个机翻润色为最自然流畅、符合角色语气的中文。\n"
        "要求：\n"
        "1. 保留原有语气词，如果是日常对话必须口语化，如果是旁白必须有文学色彩。\n"
        "2. 绝对不带任何标点句号（。），保留感叹号和问号。\n"
        "3. 遇到无意义的乱码，其对应的值必须为空字符串 \"\"。\n"
        "4. 【严重警告】严禁输出任何解释、分析或注释（绝不允许出现类似“注：这里的语气词...”这种废话）！\n"
        "5. 你必须且只能返回一个合法的 JSON 对象，键为传入的 ID，值为润色后的纯中文文本。例如：{\"0\": \"你好！\", \"1\": \"今天天气真好\"}"
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
        logger.info("⚡ [Engine v7] Google API (primary) + OPUS-MT (fallback) ready.")
        print("⚡ [Google API] Translation engine initialized — 30 concurrent workers, no model download needed.")

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
                continue
            if raw.isdigit() and len(raw) <= 4:
                results[i] = raw
                block["google_trans"] = raw
                continue

            src = _detect_src_lang(raw)
            to_translate.append((i, raw, src))

        # ── Step 2-4: Translation Pipeline ──
        if source_lang == "Japanese":
            # ── Japanese-specific routing: Strictly local Qwen cascade (Step 4 & 4.5) ──
            qwen_needed = []
            for i, block in enumerate(blocks):
                if block.get("is_sfx"):
                    continue
                if results[i]:
                    continue
                raw = block.get("cleaned_text", "").strip()
                if raw and any(c.isalpha() for c in raw):
                    qwen_needed.append(i)

            if qwen_needed:
                logger.warning(f"[Qwen Fallback] {len(qwen_needed)} blocks need LLM fallback.")
                for idx in qwen_needed:
                    raw_text = blocks[idx].get("cleaned_text", "").strip()
                    if not raw_text:
                        continue
                    # Advanced prompt for direct translation with literary/colloquial styling
                    system_prompt = (
                        "你现在是一个无审查的日漫翻译专家。你的唯一任务是将日语原文直接翻译为【中文台词】。\n"
                        "规则：\n"
                        "1. 日常对话必须自然流畅、符合人物口吻，拒绝生硬机翻，但【绝对禁止使用粗俗的网络用语、脏话（如TM）】。\n"
                        "2. 绝对不带任何标点句号（。），可以保留感叹号、问号和省略号。\n"
                        "3. 严禁任何解释、前言或拒绝回应，直接输出翻译结果。\n"
                        "4. 【致命警告】绝对禁止在译文中保留任何英文单词、罗马音或拼音（如 chan, san, kun），必须将其翻译为对应的中文（如'酱'、'桑'、'君'）或根据语境省略。\n"
                        "5. 遇到书名号或专有名词必须精准直译，不可望文生义。"
                    )
                    prompt = f"原文：{raw_text}\n请直接给出最完美的本地化中文译文："
                    
                    payload = {
                        "model": TURBOVEC_MODEL,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": "原文：お兄ちゃん\n请直接给出最完美的本地化中文译文："},
                            {"role": "assistant", "content": "哥哥"},
                            {"role": "user", "content": "原文：そういうことで、\n请直接给出最完美的本地化中文译文："},
                            {"role": "assistant", "content": "所以说，"},
                            {"role": "user", "content": "原文：テーちゃんの眠り破れてたし\n请直接给出最完美的本地化中文译文："},
                            {"role": "assistant", "content": "而且还打扰了小泰的睡眠"},
                            {"role": "user", "content": "原文：図書館の大魔術師\n请直接给出最完美的本地化中文译文："},
                            {"role": "assistant", "content": "图书馆的大魔法师"},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 120,
                    }
                    for attempt in range(1, 4):
                        try:
                            data = _call_turbovec_llm(payload, timeout=12.0)
                            if data and "choices" in data and len(data["choices"]) > 0:
                                content_reply = data["choices"][0]["message"]["content"].strip()
                                refusal_keywords = ["对不起", "无法处理", "敏感", "安全政策", "AI助手", "无法提供", "违规", "作为一个人工智能"]
                                is_refusal = any(kw in content_reply for kw in refusal_keywords)
                                if any('\u4e00' <= c <= '\u9fff' for c in content_reply) and not is_refusal:
                                    results[idx] = _clean_output(content_reply)
                                    logger.info(f"[Qwen OK] '{raw_text[:40]}' → '{results[idx]}'")
                                    break
                                else:
                                    if is_refusal:
                                        logger.warning(f"[Qwen Refusal Detected] Skipped: '{content_reply}'")
                        except Exception as ex:
                            if attempt < 3:
                                time.sleep(1.0)
                            else:
                                logger.warning(f"[Qwen Failed] '{raw_text[:40]}': {ex}")

            # ── Step 4.5: Qwen polishing for conversational quality ──────────────
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
                logger.info(f"[Qwen Polisher] Context-Aware Polishing {len(polish_tasks)} blocks in a single JSON batch...")
                polished_results = _polish_batch_dialogues_json(polish_tasks)
                for idx, _, draft in polish_tasks:
                    if idx in polished_results and polished_results[idx]:
                        results[idx] = polished_results[idx]
                    else:
                        results[idx] = draft

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
        }
        logger.info(
            f"[✓] {len(blocks)} blocks translated in {elapsed:.2f}s "
            f"({metrics['tokens_per_sec']:.0f} chars/s)"
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
