# 功能说明

### [Home] 流水线模式切换（stream / overlap / serial）
- 说明：用来 A/B 对比三种调度方式的速度。完成时会报告各阶段的完成时刻。
  - `stream`（默认，原有行为）：提取、翻译、重绘三段同时进行。
  - `overlap`（模式1）：提取和翻译并行；提取结束后卸载 OCR 模型（释放 RAM/VRAM），再开始重绘，重绘期间翻译继续进行。
  - `serial`（模式2）：先提取全部页 → 卸载 OCR → 翻译全部页（大上下文：多页台词合进一次 LLM 调用，默认 36 行）→ 最后重绘全部页。
- 涉及文件/模块：
  - `pdf_translate/main.py`：`translation_worker` 中的 `translate_pages / run_serial / unload_ocr_models / stage_times`；上传接口新增 `pipeline_mode`、`context_chunk_size` 参数。
  - `pdf_translate/core/async_stages.py`：新增 `hold_render_until_source_done`、`on_source_done` 参数。
  - `pdf_translate/core/engine.py`：`translate_batch(context_chunk_size=)`、`_batch_max_tokens`、`_batch_timeout`，失败的大批次会自动对半拆分重试。
  - `pdf_translate/templates/index.html`：新增"流水线模式"下拉框，完成时在控制台打印各阶段时刻。
- 实现要点：
  - `overlap` 模式下，翻译结果队列不限长，这样翻译不会被还没启动的重绘卡住。
  - `serial` 模式按整页分组，每组不超过 `context_chunk_size` 行。
  - 大批次失败时（例如超出服务器上下文长度）会拆成两半重试，最小拆到 12 行，然后才走单句 / Google fallback，避免整批降级。
  - 输出预算随批次大小增长：`max_tokens` = 64×行数+256（范围 2048-4096），超时 = max(18s, 1.2×行数+10s)。
  - 卸载 OCR 之后，下一个任务需要重新加载模型（首页会慢一些）。
- 相关测试：
  - `tests/test_async_stages.py`：overlap 相关 2 个用例。
  - `tests/test_context_batch.py`：4 个用例。
  - stub 版依赖跑 `main.translation_worker`，三种模式的事件顺序都符合设计，取消也正常。
  - 真实速度对比：`scratch/bench_modes.py`（需在 Windows 上跑）。
- 实测（2026-09-28，第5巻 1-10 页，RTX 3050 4GB + 12 线程）：serial 109.2s / stream 129.7s。两者都已修复 B13，翻译 0 失败；serial 只用 1 次 LLM 调用翻完 26 句。
- 默认模式：serial（`DEFAULT_PIPELINE_MODE`，可用环境变量 `PIPELINE_MODE` 覆盖；前端下拉框默认选中 serial）。
- 状态：Done

### [全局] 提取模型按剩余显存自动上 GPU
- 说明：每个任务开始时探测剩余显存（LLM 已占用的部分不算在内）。放得下的提取模型放到 GPU，放不下的留在 CPU。运行中遇到 GPU OOM / 设备错误时，该模型自动降级到 CPU 并重试当前这一步。
- 涉及文件/模块：
  - `pdf_translate/core/gpu_budget.py`（新增）：显存探测 + 分配计划。
  - `pdf_translate/core/extractor.py`：`prepare_extract_devices`、`_device_for`、`_demote_to_cpu`；三个模型的 getter 按分配结果选择设备。
  - `main.py` / `core/pipeline.py`：每个任务开始时调用；完成信息里带 `extract_devices`，前端控制台会显示。
  - `scratch/gpu_diag.py`：环境诊断脚本。
- 实现要点：
  - 显存探测顺序：`torch.cuda.mem_get_info` → `nvidia-smi` → Windows WMI GPU 计数器。用 WMI 类名而不是 Get-Counter 路径，是因为后者在中文 Windows 上会被本地化。
  - 显存估算：CTD 700MB、MangaOCR 900MB、PaddleOCR 300MB，按这个顺序优先分配。预留 `EXTRACT_VRAM_RESERVE_MB`（默认 400MB）给 LLM 的 KV cache 增长和桌面使用。
  - 已经在 GPU 上的模型有迟滞：剩余显存 ≥ 预留的一半就继续留在 GPU，避免每个任务来回切换。
  - 后端：ORT 模型用 CUDA EP 或 DirectML EP；MangaOCR 需要 CUDA 版 PyTorch，CPU 版 torch 下永远留在 CPU。
  - 环境变量 `EXTRACT_DEVICE=auto|cpu|gpu` 可手动覆盖。
  - 自动校准：模型第一次上 GPU 时，测量「加载 + 首次推理」前后的剩余显存差 ×1.2，写入 `data/gpu_model_vram.json`（本机专用，已加入 gitignore）。之后的任务按实测值分配，不再用估算值。
  - 行为变化：以前只要装了 DirectML，CTD 就一律上 GPU。现在如果探测不到剩余显存，会留在 CPU（保守默认）。
- 相关测试：`tests/test_gpu_budget.py`（13 个用例），已加入 pre-commit。
- 状态：Done（待用户在 Windows 上跑 `scratch/gpu_diag.py` 确认探测方式）

### [全局] 重绘并行 + 重绘耗时分解
- 说明：serial 模式下重绘不再一页一页串行跑，同时开 `RENDER_CONCURRENCY` 页（默认 3）。LaMa 允许 `LAMA_PARALLEL` 个 crop 同时推理（逻辑核数 ≥8 时默认 2，否则 1），每个推理用 `LAMA_THREADS` 线程（默认 逻辑核数/2/LAMA_PARALLEL，至少 2）。完成时报告 `render_breakdown`：rasterize / masks / telea / lama_page / lama_run / lama_wait / draw_save / page_total 各自的调用次数和秒数。
- 涉及文件/模块：`core/renderer.py`（`_LAMA_RUN_SEM`、`_rstat`、`get_render_stats`）、`main.py`（`run_serial` 并发重绘、`render_breakdown`）、`scratch/bench_modes.py`。
- 实现要点：ORT 的 `InferenceSession.run` 本身线程安全，原来的全局锁换成了 BoundedSemaphore。背景：用户实测 serial 模式重绘 106s（10 页）是瓶颈。在 VM（2 核）上 4 个 crop 的测试：LAMA_PARALLEL 1 → 28.2s，2 → 19.8s。
- 相关测试：`tests/test_render_async.py` 新增 2 个用例（耗时分解、多页 CPU 阶段确实重叠）。
- 状态：Done（待用户实测）

### [实验] LaMa 用 DirectML 跑 GPU —— 不可行
- 结果（2026-09-28，RTX 3050 Laptop，已停 LLM）：DML session 能创建，但第一次推理时，FFC 的 `FourierUnit ... Transpose_56` 节点报 `887A0005 The GPU device instance has been suspended`（GPU 设备被挂起）。这与 renderer 里"LaMa 在 CPU 上跑以保证 FFC DFT 稳定"的原注释一致。
- 处理：生产代码保持 LaMa 只走 CPU（原本就如此，没有改动）。测试脚本 `scratch/lama_gpu_test.py` 保留，以后换 CUDA EP 时可以复用。

### [实验] LaMa 用 CUDA EP（onnxruntime-gpu）跑 GPU —— 不可行
- 结果（2026-09-28，RTX 3050 Laptop 4GB，已停 LLM，隔离环境 `.venv-cuda`）：
  - 默认图优化下，session 创建直接失败：`DFT ... one-sided DFT requires real input`。
  - 改成 `ORT_DISABLE_ALL` 后能创建 session，但第 1 块区域的预热就花了 280.8 秒（CPU 每块约 6-8 秒）。预热期间显存一路涨到 3955MB（显存满了），GPU 使用率只有 10-44%，之后单块推理也超过 20 秒。
- 结论：这个 LaMa ONNX 导出在 4GB 显卡的 CUDA 上跑，比 CPU 慢 30 倍以上，放弃。生产代码不受影响（LaMa 一直都在 CPU 上）。
- 可以清理：`.venv-cuda\` 文件夹（约 2GB，已加入 gitignore）可以手动删掉。

### [实验] LaMa 多区域拼图（一次调用处理多块）—— 暂不采用
- 背景：`lama.onnx` 输入固定为 512x512，没办法用更小的推理尺寸；唯一能减少调用次数的办法是把多块小区域拼进同一张 512 的图里。
- 结果（用户实测，第5巻 1-20 页，在 B15 修复之前）：LaMa 调用从 20 次减到 16 次，耗时从 143.9s 降到 117.9s（快 1.22 倍）。但拼图结果和原方式的差异较大：mask 内 PSNR 在 10.8-31.8 dB 之间，12 页里有 9 页低于 25 dB。原因是 FFC 的感受野覆盖整张图，同一张图里的其他小块会影响补出来的内容。
- 结论：只省下约 18% 的 LaMa 时间（10 页约 10s），画质却变得不可预测，所以不采用。脚本保留在 `scratch/lama_pack_test.py`。
- B14 + B15 修复后的实测（第5巻 1-10 页，serial 模式）：总耗时 109.2s（与之前持平）；LaMa 调用 14 → 12 次，CPU 计算时间 115s → 101s；提取阶段 28.1s → 35.6s，属于测量波动或模型冷启动；消字质量明显改善。

### [全局] stream / overlap 模式按行数攒批翻译
- 说明：stream / overlap 模式原来每提取完一页就单独翻译一页（每批最多 12 行，上下文很小）。现在先把提取好的页攒起来，累计到 `translate_batch_lines` 行（默认 36，和 serial 的每批行数一致）或者提取全部结束时，才一次性翻译这些页。LLM 能看到跨页的对话，调用次数也更少。
- 涉及文件/模块：`core/async_stages.py`（`translate_batch` / `batch_lines` / `page_lines` 参数、`batched_translator`）、`main.py`（`DEFAULT_STREAM_BATCH_LINES`，环境变量 `TRANSLATE_BATCH_LINES`，上传参数 `translate_batch_lines`；设为 0 就回到原来每页翻译一次）、`scratch/bench_modes.py --batch-lines`。
- 实现要点：只攒整页，不把一页拆到两批里。已有缓存的页照常跳过。出错时仍然走 abort，不会死锁。代价是重绘要等第一批（约 36 行，大约 4-8 页）翻译完才能开始。
- 相关测试：`tests/test_async_stages.py` 新增 2 个用例（按行数分批、跳过缓存页并正确传出错误）。用 stub 依赖跑 main：每页 10 行、阈值 36 时，stream 和 overlap 都是每 4 页翻译一次（t1234 / t5678），serial 不受影响。
- 状态：Done（待实测）

### [全局] 汉化组风格：译文用词语气 + 字体字号
- 说明：让输出更像普通的中文汉化版日漫。
- 用词语气（`core/engine.py`）：日语批量翻译改用 `MANGA_SYSTEM_PROMPT`，定位是汉化组翻译兼嵌字编辑。要求口语化短句、长度接近原文；用中文语气词体现 ね/よ/な/ぞ；称谓本土化；拟声词译成中文象声词；附 4 条示例。`_manga_punct()` 统一标点：不用句号，省略号统一为"……"，"!?"→"！？"，"~"→"～"，句尾不留逗号。省略号在 `_clean_output` 最前面就转换，避免后面"去掉句尾点号"的步骤把它删掉。
- 字体（`core/renderer.py`）：优先使用常规字重，不再默认用微软雅黑粗体（msyhbd），粗体只作为最后的备选。可以用环境变量 `MANGA_FONT=<路径>` 指定字体，或把 .ttf/.otf/.ttc 放进 `pdf_translate/fonts/`（例如汉仪中圆、方正准圆、思源黑体 Medium），会优先使用。
- 字号：`_estimate_glyph_px()` 用 CTD 的文字分割估算原文字号（竖排取列宽中位数，横排取行高中位数，忽略振假名这类细条），译文字号上限 = 1.05 × 原文字号，不再把气泡塞满大字。
- 描边：干净气泡里不再给字加白边（以前加了会显得很粗）；画面、网点上的字保留细描边（fs/14），保证看得清。
- `RENDER_CACHE_VERSION` 已更新，旧的页面缓存不会被复用。
- 相关测试：`tests/test_manga_style.py`（5 个用例），已加入 pre-commit。示意图：`scratch/style_p11.jpg`（原图 | 新排版；图中译文是随便放的示例句）。
- 状态：Done（待用户实测）

### [全局] 三种模式实测对比（2026-09-28，第5巻 1-10 页，B14-B17 与攒批翻译之后）
| 模式 | 总耗时 | 提取完成 | 翻译完成 | 说明 |
|---|---|---|---|---|
| serial | **95.5s** | 26.8s | 37.8s | 重绘时同时处理 3 页 |
| overlap | 131.9s | 25.6s | 37.5s | 重绘只能一页一页来 |
| stream | 153.9s | 33.8s | 45.1s | 提取时和 LLM 抢资源；重绘一页一页来 |
- 三种模式都只调用了 1 次 LLM（26 句不到攒批阈值 36 行，所以一次翻完），0 失败。
- 差距主要在重绘：stream 和 overlap 的重绘 worker 只有一个。所以 `run_three_stage_pipeline` 新增了 `render_concurrency`，main 对所有模式都传 `RENDER_CONCURRENCY`（默认 3）。预计 overlap 能接近 serial。
- 默认模式保持 serial。

### [全局] 默认模式改为 auto（按页数自动选择）
- 实测（2026-09-28，第5巻 1-50 页，用户机器）：
  - overlap 472.1s：提取 331.0s（LLM 同时在跑，所以提取慢一些），翻译在提取结束后 15s 就完成，重绘 138s。
  - serial 566.5s：提取 281.0s，之后翻译 147s，重绘 136s。跑 serial 时用户同时在用 Claude 改代码，结果可能偏慢。
  - 结论：页数一多，翻译时间（约 145s LLM）就值得藏到提取后面同时进行；只有 10 页时翻译约 10s，没什么可藏的，serial 更快（95.5s）。两种模式的重绘时间现在基本一样（渲染并发已修）。
- 改动：`PIPELINE_MODES` 加入 `auto`，`resolve_pipeline_mode()`：少于 `AUTO_OVERLAP_MIN_PAGES`（默认 20，可用环境变量调整）页用 serial，否则用 overlap。默认 `PIPELINE_MODE=auto`，前端下拉框默认选中"自动"。完成信息里会显示成类似 `auto->overlap`。
- 相关测试：`tests/test_low_bugs.py::test_auto_mode_*`。用 stub 依赖跑：8 页 → serial 流程；把阈值设为 5 → overlap 流程。

### [全局] MangaOCR 批量识别（每页几次 generate，代替每个框一次）
- 说明：50 页里提取约占总时间的 70%，其中 MangaOCR 在 CPU 上每个文本框约 1 秒。现在同一页的所有框先裁好图，按 `MANGA_OCR_BATCH`（默认 8）一批，堆叠后调一次 `model.generate()`。每个框仍然是单独识别的（只是一起送进模型），结果按框存进 `manga_cache`，后续的 OCR 链、写入块、重绘都用原来的框和位置（用户要求：重绘要回到原本的位置）。
- 涉及文件/模块：`core/extractor.py`（`_manga_ocr_batch`、`_extract_single_page` 里的 2b 步骤）、`scratch/mangaocr_batch_test.py`（在真实裁图上对比逐个识别和批量识别的耗时，以及文字是否一致）。
- 实现要点：批量识别失败时自动退回逐个识别；某个框结果为空时照常走 PaddleOCR / EasyOCR fallback。ONNX 方案（`manga_ocr_encoder.onnx`）只导出了 encoder，在 VM 上加载 345MB 的外部数据文件就超时了，这次先不做。
- 相关测试：`tests/test_manga_batch.py`（2 个用例），已加入 pre-commit。
- 状态：Done（待用户用脚本实测加速倍数）

### [Extract] 提取耗时分解 (extract breakdown)
- 说明：统计每页提取各阶段耗时，找出真正瓶颈；MangaOCR batch 默认改为 16（实测 0.35→0.22s/crop，文本一致 13/13）。
- 涉及文件/模块：core/extractor.py（_xstat / reset_extract_stats / get_extract_stats）、main.py（metrics.extract_breakdown、SSE complete）、scratch/bench_modes.py。
- 实现要点：阶段 rasterize / load_detector / detect / load_ocr_models / manga_ocr_batch / ocr_per_box / page_total；线程安全累加。
- 相关测试：tests/test_extract_stats.py
- 状态：Done
