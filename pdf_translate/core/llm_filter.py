import httpx
import json
import logging
import re
import time
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class LLMTextPreprocessor:
    """
    Passes OCR text blocks to a local LLM (e.g., Qwen2.5) to:
    1. Correct OCR typos/broken words.
    2. Classify blocks as Sound Effects (SFX) or dialogue.
    """
    def __init__(self, api_base="http://localhost:18088/v1", model_name="docker.io/ai/qwen3.5:4b-q4_K_M"):
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.client = httpx.Client(timeout=90.0)
        
    def _build_prompt(self, blocks: List[Dict[str, Any]]) -> str:
        prompt_data = []
        for b in blocks:
            item = {
                "id": b["id"],
                "english": b.get("cleaned_text", b.get("text", "")).strip(),
                "google_trans": b.get("google_trans", "").strip()
            }
            if "page_num" in b:
                item["page"] = b["page_num"]
            prompt_data.append(item)
            
        system_prompt = (
            "You are a professional manga translation editor, translator, and SFX (sound effect) detector.\n"
            "You are given a JSON array of text blocks from one or more manga pages.\n"
            "Each block contains the original OCR 'english' text and a baseline 'google_trans'.\n"
            "Manga Context Hints (\"Magus of the Library\" / \"圕的大魔法师\" series proper nouns & official Chinese translations):\n"
            "- Kafna -> 卡夫那 (librarians/司书)\n"
            "- Kohkwa -> 库瓦 (assistants to Kafna/司书助手)\n"
            "- Sae Fumis -> 萨埃·芙蜜斯 (Sae -> 萨埃)\n"
            "- Tepel Huracaan -> 特佩尔·乌拉坎 (Tepel -> 特佩尔)\n"
            "- Theo -> 西奥\n"
            "- Sedona -> 塞多纳\n"
            "- Medina -> 麦地那\n"
            "- Amun -> 阿蒙\n"
            "- Hyron -> 亥隆 (race)\n"
            "- Kadira -> 卡迪拉 (race)\n\n"
            "Your tasks for each block:\n"
            "1. Editor Review & Translation (Simplified Chinese - 华文):\n"
            "   - You MUST fill the 'trans' field with a natural, polished Simplified Chinese translation.\n"
            "   - If 'google_trans' is empty, is in English, or has translation errors, you MUST translate the original 'english' text into Simplified Chinese yourself.\n"
            "   - Do NOT leave 'trans' as English under any circumstances, unless the text is a number or symbol.\n"
            "   - Apply the proper nouns from the Context Hints (e.g. change 'Kohkwa' to '库瓦', 'Theo' to '西奥').\n"
            "   - ABSOLUTELY NO CONVERSATIONAL REFUSALS: If the text is illegible or you cannot translate it, NEVER apologize or say 'I cannot translate this' (e.g. '对不起，我不理解'). Instead, treat it as noise: set 'sfx': true and 'trans': \"\".\n"
            "   - DO NOT HALLUCINATE: If OCR text is random noise or disconnected letters (e.g. '4T HVP', 'MRNCH'), do NOT invent random Chinese words (like '免疫球蛋白'). Treat it as noise: set 'sfx': true and 'trans': \"\".\n"
            "2. Sound Effect (SFX) & Noise Detection:\n"
            "   - Classify block as `sfx: true` ONLY if it is a background sound effect (e.g. 'YAAAWN', 'MRNCH', 'KRNGH', grunts, garbage noise).\n"
            "   - If `sfx: true`, set `trans` to \"\".\n"
            "\n"
            "Return your response strictly as a JSON object containing a list of objects under the key 'blocks'. Do not add markdown or explanations outside the JSON.\n"
            "The JSON object must look exactly like this:\n"
            "{\n"
            "  \"blocks\": [\n"
            "    {\n"
            "      \"id\": 1,\n"
            "      \"trans\": \"polished/corrected Chinese text\",\n"
            "      \"sfx\": false\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Each object in the array must contain exactly:\n"
            "  - 'id': integer id from input.\n"
            "  - 'trans': polished Chinese text.\n"
            "  - 'sfx': boolean.\n\n"
            "Input:\n"
            f"{json.dumps(prompt_data, indent=2)}"
        )
        return system_prompt

    def _extract_json_from_response(self, text: str) -> List[Dict]:
        text = text.strip()
        # Handle markdown blocks if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
            
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                # Check if any value is a list (e.g. {"blocks": [...]})
                for val in parsed.values():
                    if isinstance(val, list):
                        return val
                # Otherwise, it might be a single dictionary representing one block
                return [parsed]
            return []
        except json.JSONDecodeError:
            logger.error(f"[LLM] Failed to parse JSON response: {text[:100]}...")
            return []

    def process(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not blocks:
            return blocks
            
        system_prompt = self._build_prompt(blocks)
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You output only valid JSON."},
                {"role": "user", "content": system_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
            "options": {
                "num_predict": 2048
            },
            # Force JSON format if supported by the model/API
            "response_format": {"type": "json_object"} 
        }
        
        max_attempts = 3
        response = None
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"[LLM] Sending {len(blocks)} blocks to {self.model_name} for OCR translation (attempt {attempt}/{max_attempts})...")
                response = self.client.post(f"{self.api_base}/chat/completions", json=payload)
                response.raise_for_status()
                break
            except Exception as e:
                if attempt == max_attempts:
                    raise e
                logger.warning(f"[LLM] Attempt {attempt} failed: {e}. Retrying in 2 seconds...")
                time.sleep(2.0)
        
        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            corrected_data = self._extract_json_from_response(content)
            
            # Create a lookup mapping for easy assignment
            correction_map = {item["id"]: item for item in corrected_data if "id" in item}
            
            for block in blocks:
                b_id = block["id"]
                if b_id in correction_map:
                    corr = correction_map[b_id]
                    old_text = block.get("text", "")
                    translated_text = corr.get("trans", "").strip()
                    is_sfx = corr.get("sfx", False)
                    
                    block["original_ocr_text"] = old_text
                    block["translated_text"] = translated_text
                    block["is_sfx"] = is_sfx
                else:
                    block["is_sfx"] = False
                    block["translated_text"] = ""
                    
        except Exception as e:
            logger.warning(f"[LLM] Post-processing failed (is Ollama running?): {e}. Falling back to raw OCR text.")
            for block in blocks:
                block["is_sfx"] = False
                block["translated_text"] = ""
                
        return blocks
