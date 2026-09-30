# Third-party components / 第三方组件与模型出处

This project (MIT, see `LICENSE`) **does not include or redistribute** any of the models or libraries below.
They are downloaded / installed by the user (see README) and remain under their own licenses.
本项目（MIT）**不包含、也不分发**下列任何模型或库，均由使用者自行下载 / 安装，各自保留原许可证。

## Models

| Component | Used for | License | Source | Checked |
|---|---|---|---|---|
| comic-text-detector (ONNX) | detect speech bubbles / text mask | GPL-3.0 | https://github.com/dmMaze/comic-text-detector | yes (repo page) |
| manga-ocr + manga-ocr-base | Japanese manga OCR | Apache-2.0 | https://github.com/kha-white/manga-ocr · https://huggingface.co/kha-white/manga-ocr-base | yes |
| EasyOCR | fallback OCR | Apache-2.0 | https://github.com/JaidedAI/EasyOCR | yes |
| RapidOCR (PaddleOCR models, ONNX) | fallback OCR | Apache-2.0 | https://github.com/RapidAI/RapidOCR | yes |
| LaMa / big-lama | inpainting (text removal) | Apache-2.0 (weights page); original repo license not read | https://github.com/advimman/lama · https://huggingface.co/smartywu/big-lama | weights page only |
| Qwen3.5-4B | local translation | Apache-2.0 | https://huggingface.co/Qwen/Qwen3.5-4B | yes (model card) |
| Sakura-13B (optional) | optional ACG translation model | see its repository | https://github.com/SakuraLLM/Sakura-13B | no |

## Libraries

| Component | License | Note |
|---|---|---|
| PyMuPDF (`fitz`) | **AGPL-3.0** or Artifex commercial license | see the note below |
| FastAPI, openpyxl, onnxruntime | MIT | not re-checked in this session |
| OpenCV, Pillow, NumPy, SciPy, scikit-image, PyTorch | Apache-2.0 / HPND / BSD-style | not re-checked in this session |

### Note on PyMuPDF (AGPL-3.0)

The code of this project is MIT, and PyMuPDF is installed by the user (`pip install`), not bundled here.
If you **redistribute a bundle that contains PyMuPDF**, or **offer this program as a network service to others**,
the AGPL-3.0 terms of PyMuPDF may apply to what you distribute / offer (or you need an Artifex commercial license).
For personal, local use nothing extra is required. This is not legal advice.
若把 PyMuPDF 一起打包分发，或把本程序作为网络服务提供给他人，PyMuPDF 的 AGPL-3.0 条款可能适用于你分发 / 提供的内容（或需向 Artifex 购买商业授权）；个人本地使用无额外要求。以上不构成法律意见。

## Design references (no code included)

- [zyddnys/manga-image-translator](https://github.com/zyddnys/manga-image-translator) (GPL-3.0): inspired the bubble text-box handling and typesetting ideas. Its source is **not** part of this repository.
  `pdf_translate/export_onnx.py` can use a separate checkout of it (set `MIT_SOURCE_DIR`).

## Fonts and content

- No fonts are bundled. The renderer uses a font installed on the system, or one you put in `pdf_translate/fonts/` / `MANGA_FONT`; check that font's license before redistributing output.
- No manga pages are included. Only process files you legally own.
