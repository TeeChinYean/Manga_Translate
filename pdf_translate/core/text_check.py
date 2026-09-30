"""
Post-translation language check (target: Simplified Chinese).

The LLM step only requires "at least one CJK character" in a reply, so a translation can still
carry kana, Hangul, Cyrillic, stray Latin words or decoration symbols (※ ◆ * # ...). This module
finds those, and `check_and_fix` repairs them:

  hard issues  foreign script / kana / Latin words that are not in the source text
               -> the line is re-translated (max `max_retries`, with a hint naming the offenders),
                  and if it is still bad the offending characters are removed
  soft issues  decoration / stray symbols -> removed directly (a retry would not help)

Emotive symbols are always kept (！？…～♪♡♥★☆ and emoji), and so are the punctuation marks a
Chinese sentence needs (，。、：；—「」『』“”‘’（）《》·). Everything else counts as a symbol.
If nothing Chinese is left after cleaning, the text becomes "" so the renderer keeps the original
bubble (same rule as an untranslated line).
"""
import re
import unicodedata

# ── What may stay ────────────────────────────────────────────────────────────
EMOTIVE = set("！？…～‼⁉♪♫♬♡♥❤❣★☆✨♩")          # feelings / tone marks
SENTENCE_PUNCT = set("，。、：；—―·・「」『』“”‘’（）《》〈〉 　")
KEEP = EMOTIVE | SENTENCE_PUNCT
# ASCII punctuation is not an issue: it is converted to its Chinese form by `sanitize`
ASCII_TO_ZH = {",": "，", ".": "。", ":": "：", ";": "；", "!": "！", "?": "？", "~": "～",
               "(": "（", ")": "）", '"': "”", "'": "’", "-": "—"}
# Allowed next to a digit (50%, 3.5, 12:30, 1/2, +3)
DIGIT_SYMBOLS = set("%.:/-+×÷,")

_EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF️‍]")
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z']*")
_MULTISPACE = re.compile(r"[ 　]{2,}")


def is_chinese_target(target_lang: str) -> bool:
    t = (target_lang or "").lower()
    return "chinese" in t or t.startswith("zh") or "中文" in t


def _script_of(ch: str) -> str:
    """'kana' | 'hangul' | 'other-script' | 'latin' | 'cjk' | 'digit' | 'symbol' | 'space'"""
    o = ord(ch)
    if 0x3040 <= o <= 0x30FF and ch not in "・" or 0xFF66 <= o <= 0xFF9F or 0x31F0 <= o <= 0x31FF:
        return "kana"
    if 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F:
        return "hangul"
    if 0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF or 0x20000 <= o <= 0x2FA1F:
        return "cjk"
    if ch.isspace():
        return "space"
    if ch.isdigit():
        return "digit"
    cat = unicodedata.category(ch)
    if cat[0] == "L":
        return "latin" if (ch.isascii() or "LATIN" in unicodedata.name(ch, "")) else "other-script"
    return "symbol"


def _source_words(source: str) -> set:
    return {w.lower() for w in _LATIN_WORD.findall(source or "")}


def _scan(text: str, source: str):
    """Yield (index, char, kind, severity) for every offending character.
    kind: kana / hangul / script / latin / symbol; severity: 'hard' | 'soft'."""
    src_words = _source_words(source)
    n = len(text)

    # Latin words are judged as a whole: fine if the source contains the same word (OK, HP ...)
    latin_ok = set()
    for m in _LATIN_WORD.finditer(text):
        if m.group().lower() in src_words:
            latin_ok.update(range(m.start(), m.end()))

    for i, ch in enumerate(text):
        if ch in KEEP or _EMOJI.match(ch):
            continue
        kind = _script_of(ch)
        if kind in ("cjk", "digit", "space"):
            continue
        if kind == "kana":
            yield i, ch, "kana", "hard"
        elif kind == "hangul":
            yield i, ch, "hangul", "hard"
        elif kind == "other-script":
            yield i, ch, "script", "hard"
        elif kind == "latin":
            if i not in latin_ok:
                yield i, ch, "latin", "hard"
        else:  # symbol / punctuation
            if ch in ASCII_TO_ZH:
                continue
            if ch in DIGIT_SYMBOLS:
                near = (i > 0 and text[i - 1].isdigit()) or (i + 1 < n and text[i + 1].isdigit())
                if near:
                    continue
            yield i, ch, "symbol", "soft"


def find_issues(text: str, source: str = "") -> dict:
    """{"hard": [chars], "soft": [chars]} (unique, in order). Empty lists = the text is fine."""
    hard, soft = [], []
    for _, ch, _, sev in _scan(text or "", source):
        bucket = hard if sev == "hard" else soft
        if ch not in bucket:
            bucket.append(ch)
    return {"hard": hard, "soft": soft}


def has_issues(text: str, source: str = "") -> bool:
    r = find_issues(text, source)
    return bool(r["hard"] or r["soft"])


def sanitize(text: str, source: str = "") -> str:
    """Remove every offending character, convert ASCII punctuation to Chinese punctuation.
    Returns "" when no Chinese character is left."""
    text = text or ""
    bad = {i for i, _, _, _ in _scan(text, source)}
    out = []
    for i, ch in enumerate(text):
        if i in bad:
            continue
        if ch in ASCII_TO_ZH and _script_of(ch) == "symbol":
            near_digit = (i > 0 and text[i - 1].isdigit()) and (i + 1 < len(text) and text[i + 1].isdigit())
            out.append(ch if near_digit and ch in DIGIT_SYMBOLS else ASCII_TO_ZH[ch])
        else:
            out.append(ch)
    s = "".join(out)
    s = _MULTISPACE.sub(" ", s)
    s = re.sub(r"([，。、：；])\1+", r"\1", s)              # 。。 -> 。
    s = re.sub(r"^[，。、：；—\s]+|[，、：；—\s]+$", "", s)   # dangling punctuation left by removals
    s = s.strip()
    if not any(_script_of(c) == "cjk" for c in s):
        return ""
    return s


def retry_hint(issues: dict) -> str:
    """Extra instruction appended to the LLM prompt when a line is re-translated."""
    chars = "".join(issues.get("hard", []) + issues.get("soft", []))[:12]
    return ("【上一次译文含有不允许的字符：" + chars + "。请只用简体中文和常规中文标点重新翻译，"
            "不要保留日文假名、外文字母或特殊符号；可以保留 ！？…～♪♡ 这类语气符号。】")


def check_and_fix(entries: list, retranslate, source_lang: str = "Japanese",
                  target_lang: str = "Simplified Chinese", max_retries: int = 2) -> list:
    """
    entries: [{"raw": <source text>, "text": <translation>, ...extra keys are kept}]; "text" is
             updated in place.
    retranslate(bad_entries, hint) -> list[str] | None: new translations for `bad_entries`
             (same order). May raise; a failed retry just counts as "still bad".
    Returns one report dict per entry that needed work:
        {"entry": entry, "before": str, "after": str, "retries": int,
         "action": "retranslated" | "cleaned" | "emptied", "removed": "chars"}
    Nothing is checked unless the target language is Chinese.
    """
    if not is_chinese_target(target_lang):
        return []

    state = {}   # id(entry) -> report
    for e in entries:
        t = e.get("text") or ""
        if t and has_issues(t, e.get("raw", "")):
            state[id(e)] = {"entry": e, "before": t, "retries": 0, "removed": "", "issues": find_issues(t, e.get("raw", ""))}

    # 1) re-translate lines with hard issues
    for attempt in range(1, max_retries + 1):
        todo = [s["entry"] for s in state.values()
                if find_issues(s["entry"]["text"], s["entry"].get("raw", ""))["hard"]]
        if not todo or retranslate is None:
            break
        hard = "".join(dict.fromkeys(c for e in todo for c in find_issues(e["text"], e.get("raw", ""))["hard"]))
        try:
            new_texts = retranslate(todo, retry_hint({"hard": list(hard)}))
        except Exception:
            new_texts = None
        if not new_texts:
            break
        for e, nt in zip(todo, new_texts):
            state[id(e)]["retries"] = attempt
            if nt and any(_script_of(c) == "cjk" for c in nt):
                e["text"] = nt

    # 2) clean whatever is left
    report = []
    for s in state.values():
        e = s["entry"]
        cur = e["text"]
        if has_issues(cur, e.get("raw", "")):
            left = find_issues(cur, e.get("raw", ""))
            s["removed"] = "".join(left["hard"] + left["soft"])
            cleaned = sanitize(cur, e.get("raw", ""))
            e["text"] = cleaned
            s["action"] = "cleaned" if cleaned else "emptied"
        else:
            s["action"] = "retranslated" if s["retries"] else "cleaned"
            if e["text"] != s["before"] and not s["retries"]:
                s["action"] = "cleaned"
        s["after"] = e["text"]
        report.append(s)
    return report


def flag_entries(entries: list) -> list:
    """Warn-only check (user-typed text is never changed): [{"page", "id", "chars"}] for lines with issues."""
    out = []
    for e in entries:
        r = find_issues(e.get("text") or "", e.get("raw", ""))
        chars = "".join(r["hard"] + r["soft"])
        if chars:
            out.append({"page": e.get("page"), "id": e.get("id"), "chars": chars})
    return out


def build_flagged_warning(flagged: list, limit: int = 8) -> str:
    if not flagged:
        return ""
    where = [f"第{f['page']}页#{f['id']}（{f['chars'][:6]}）" for f in flagged[:limit]]
    more = f" 等共 {len(flagged)} 行" if len(flagged) > limit else ""
    return "校对译文含非简体中文字符或符号（未自动修改）: " + "、".join(where) + more


def build_language_warning(report: list, limit: int = 8) -> str:
    """One line for the completion message; "" when nothing needed fixing."""
    if not report:
        return ""
    kinds = {"retranslated": 0, "cleaned": 0, "emptied": 0}
    for r in report:
        kinds[r["action"]] += 1
    parts = []
    if kinds["retranslated"]:
        parts.append(f"{kinds['retranslated']} 行重译后合格")
    if kinds["cleaned"]:
        parts.append(f"{kinds['cleaned']} 行已清理非法字符")
    if kinds["emptied"]:
        parts.append(f"{kinds['emptied']} 行清理后无中文，保留原文")
    where = []
    for r in report[:limit]:
        e = r["entry"]
        tag = f"第{e['page']}页#{e['id']}" if "page" in e else f"#{e.get('id', '?')}"
        extra = f"（去掉「{r['removed']}」）" if r["removed"] else ""
        where.append(tag + extra)
    more = f" 等共 {len(report)} 行" if len(report) > limit else ""
    return "语言检查: " + "，".join(parts) + "：" + "、".join(where) + more
