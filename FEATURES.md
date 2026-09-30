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
- 状态：Done，默认关闭（实测：OCR 11.3s→2.2s，但每个任务启动子进程的开销让提取 19.6s→28.8s、全程 42.8s→54.1s；MANGA_OCR_GPU_WORKER=1 开启）

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
- 状态：Done，默认关闭（实测：OCR 11.3s→2.2s，但每个任务启动子进程的开销让提取 19.6s→28.8s、全程 42.8s→54.1s；MANGA_OCR_GPU_WORKER=1 开启）

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

### [Render] LaMa CPU 并行布局 3×4
- 说明：LaMa 默认改为 3 个并发 × 4 线程（12 逻辑核），实测 3.78→2.22s/crop（x1.70），输出一致；图优化级别无收益，保留 ORT_DISABLE_ALL。
- 涉及文件/模块：core/renderer.py `_default_lama_layout`；scratch/lama_speed_test.py（新增 4×3、6×2 组合）
- 实现要点：用满全部逻辑核；环境变量 LAMA_PARALLEL / LAMA_THREADS 仍可覆盖；serial 模式重绘页并发跟随 LAMA_PARALLEL。
- 相关测试：tests/test_lama_layout.py
- 状态：Done

### [Extract] OCR 模型按需常驻（OCR_UNLOAD）
- 说明：提取结束只释放占 VRAM 的模型；CPU 模型在可用 RAM ≥ 3000MB 时常驻，下一次任务省掉约 23s 冷加载。
- 涉及文件/模块：core/extractor.py `_models_to_unload` / `unload_models`；main.py 日志；requirements.txt 增加 psutil
- 实现要点：OCR_UNLOAD=auto（默认）/always（旧行为）/never；OCR_KEEP_MIN_FREE_MB 调阈值；无 psutil 时退回全部释放。
- 相关测试：tests/test_ocr_unload.py
- 状态：Done

### [Render] 同页 LaMa 区域并行
- 说明：一页有多个 LaMa 区域时并发执行（全局仍受 LAMA_PARALLEL 限制），避免"最后一页 3 个区域串行、其他核空闲"的长尾。
- 涉及文件/模块：core/renderer.py `_lama_inpaint`
- 实现要点：每个窗口用原图像素作输入（窗口内的遮罩像素本来就全部重绘），按原顺序回写；结果与串行一致。
- 相关测试：tests/test_lama_parallel_regions.py
- 状态：Done

### [Render] LaMa 预加载
- 说明：提取结束（serial：翻译进行时 CPU 空闲）就在后台加载 LaMa；重绘开始前等待加载完成，避免冷加载卡住其他页的 masks（冷启动时 masks 10 页 23–33s，热启动 0.5s）。
- 涉及文件/模块：core/renderer.py `preload_lama`、`lama_load` 计时；main.py `start_lama_preload`
- 实现要点：只加载一次（共享 future）；render_page 先 await；stream 模式在首个渲染页触发。
- 相关测试：tests/test_lama_preload.py
- 状态：Done

### [Startup] 启动预加载覆盖首个任务实际用到的模型
- 说明：启动时后台依次加载 GPU 布局、CTD、MangaOCR（含一次热身）、PaddleOCR、LaMa；EasyOCR 只是 fallback，改为按需加载。首个任务不再付 ~23s OCR + ~12s LaMa 冷加载。
- 涉及文件/模块：main.py `preload_all_models`
- 实现要点：每步单独 try/计时日志；ORT 建 session 时持有 GIL，放在启动期不会卡住任务（冷启动时翻译阶段被 LaMa 加载拖长 11s 的现象由此消除）。
- 相关测试：tests/test_lama_preload.py `test_startup_preloader_covers_primary_models`
- 状态：Done

### [Render] 气泡检测 + 字号尽量放大（B25）
- 说明：检测文字所在的对话气泡，在气泡内部可用范围里让译文尽量大且不超出；没有闭合气泡（字写在画上、开放白底）则跳过，沿用原字号逻辑。
- 涉及文件/模块：core/renderer.py `_bubble_rects`、`_layout_translations`
- 实现要点：去字后的文字框 ≥85% 为白（≥225）→ 取其所在白色连通区；区域碰到搜索窗口边界或面积 >40 倍文字框 → 视为开放区域跳过；填洞后按气泡尺寸 7% 内缩；以文字中心为轴生成 8 种高度的最大内接矩形（避开同页其他块），逐个试 `_best_font`，取字号最大者；气泡内字号上限 72px；仍保留与已排文字重叠时缩小的保护。单次检测约 60ms。
- 相关测试：tests/test_bubble_fit.py
- 状态：Done

### [Extract/Render] GPU：MangaOCR fp16 + LaMa torch CUDA（可选）
- 说明：实测（scratch/gpu_ocr_lama_check.py，RTX 3050 4GB，LLM 开着 3121MB）：MangaOCR CUDA fp16 x10–11、文字 9/9 一致，+~800MB 显存；LaMa big-lama torch CUDA 0.4s/窗口 vs ONNX CPU 6–7s（x15–17），+~600MB 显存；两者都能挨着 LLM 放下（分阶段使用，不同时）
- 涉及文件/模块：core/extractor.py（GPU 上自动 fp16，MANGA_OCR_FP16=0 关闭；输入按模型 dtype 转换）；core/gpu_budget.py（manga_ocr 预估 600MB）；core/renderer.py（`_TorchLamaSession`，LAMA_BACKEND=torch 启用，默认 onnx；找不到 CUDA / big-lama.pt 自动回退 ONNX）
- 实现要点：需要主程序的 Python 装 CUDA 版 PyTorch（目前是 +cpu）；big-lama.pt 默认从 ~/.cache/torch/hub/checkpoints 读取（simple-lama 下载的位置），或 LAMA_TORCH_PATH；torch LaMa 输出与 lama.onnx 不同（PSNR 中位 20.8 dB），需看对比图后再启用
- 相关测试：tests/test_lama_backend.py、tests/test_manga_batch.py
- 状态：Done（对比图确认画质相当或更干净 → LAMA_BACKEND 默认 auto：有 CUDA 版 PyTorch + big-lama.pt 就用 GPU，否则 ONNX CPU）

### [Extract] MangaOCR 在 GPU 子进程里跑（serial 提取阶段，B30）
- 说明：serial 模式提取开始时启动 core/manga_worker.py（CUDA fp16），提取结束（unload_models）立即结束子进程，显存全部释放后再翻译；主进程不建 CUDA 上下文（见 B29）。启动不阻塞：子进程加载时 CTD 先处理前几页，第一次 OCR 时才等待就绪
- 涉及文件/模块：core/manga_worker.py、core/extractor.py（`_MangaGpuProxy`、start/stop_manga_gpu_worker、`_manga_gpu_worker`）、main.py（run_serial）
- 实现要点：接口与 MangaOcr 一致（batch / __call__），`_manga_ocr_batch` 直接转发；子进程启动失败或超时 → 本任务回退 CPU MangaOCR；MANGA_OCR_GPU_WORKER=0 关闭；overlap/stream 模式不启用（提取与翻译同时进行，会和 LLM 抢显存）
- 相关测试：tests/test_manga_gpu_worker.py
- 状态：Done，默认关闭（实测：OCR 11.3s→2.2s，但每个任务启动子进程的开销让提取 19.6s→28.8s、全程 42.8s→54.1s；MANGA_OCR_GPU_WORKER=1 开启）

### [Render] 不用 torch 时 LaMa 改用 lama_fp32.onnx（CPU x1.83）
- 说明：scratch/lama_matrix.py 一次测完 3 个 ONNX 模型 × CPU/DirectML/CUDA × 线程/优化参数 + torch fp32/fp16（8 个窗口）：lama.onnx CPU 5.19 s/窗口 → lama_fp32.onnx 2.84 s/窗口（与 big-lama.pt 同一权重，PSNR 相同）；DirectML 全部失败；ONNX CUDA 0.29 s/窗口但 +1109MB 显存（4GB 卡挨着 LLM 放不下）；torch fp16 输出损坏（4.8 dB）→ GPU 继续用 torch fp32 子进程（+669MB）
- 涉及文件/模块：core/renderer.py（`_lama_onnx_path`、`_OnnxLamaSession`、`_get_lama_session`；RENDER_CACHE_VERSION 更新）
- 实现要点：优先 lama_fp32.onnx，没有则 lama.onnx；LAMA_ONNX_MODEL=lama-manga.onnx 可切换漫画训练版；包装层把 l_image_/l_mask_ 映射到模型实际输入名（image/mask），0..1 输出自动 ×255
- 相关测试：tests/test_lama_backend.py（名称映射 + 缩放、模型选择与 env 覆盖）
- 状态：Done

### [Pipeline] GPU LaMa 时 auto 改为 overlap：提取与翻译同时进行，重绘在翻译完成后
- 说明：用户实测「提取和翻译一起」比 serial 快。提取在 CPU，不占显存，可以和 LLM 翻译同时跑；只有 GPU LaMa 不能和正在翻译的 LLM 共用 4GB 显存（B28），所以重绘等全部翻译完成后才开始
- 涉及文件/模块：core/async_stages.py（`hold_render_until_translate_done`）、main.py（`resolve_pipeline_mode` → AUTO_GPU_MODE，overlap 开始前检查 LLM 是否被挤出显存并重启）
- 实现要点：hold 时已翻译队列不限长度，翻译不会被还没开始的重绘卡住；翻译出错时重绘不会死等；AUTO_GPU_MODE=serial 恢复旧行为；CPU LaMa 时 auto 规则不变（<20 页 serial，≥20 页 overlap）；手动选 overlap 且 LaMa 在 GPU 时同样等翻译完成再重绘
- 相关测试：tests/test_async_stages.py（重绘在所有翻译之后、翻译与提取重叠、出错不死锁）、tests/test_low_bugs.py（GPU 时 auto → overlap）
- 状态：Done。实测第5巻（scratch/bench_modes.py）：1-10 页 overlap 46.3s / serial 46.9s（只有 24 句 < 一批 36 句，翻译只能等提取完才开始）；1-30 页 overlap 173.2s / serial 244.3s（x1.41，约 90s 翻译藏进 128s 提取里，提取结束后只多等 15.5s）

### [Extract] MangaOCR CPU 参数矩阵（结论：维持现状，不改）
- 说明：scratch/mocr_matrix.py 一次测完（第5巻 1-30 页真实框，取 96 个，6 线程）：现行 torch batch16 = 0.302 s/框；batch 8 / 32 = 0.286 / 0.316；线程 4 / 8 = 0.383 / 0.351（6 最好）；ONNX encoder（导出固定 batch=1）+ torch decoder = 0.313；int8 decoder 0.278 但文字只 83/96 相同（「６月」→「８月」）→ 否决；两批同时跑 0.287
- 实现要点：所有方案差距 ±5%（测量噪声内），不值得改；encoder 占 62%（0.186 s/框），固定 224×224 输入，无法缩小。流水线里实际约 0.5 s/框，比单独测慢 1.65 倍，原因是和 CTD 同时抢 CPU（总 CPU 工作量不变，并行帮不上）
- 相关测试：无（未改代码）
- 状态：Done（测量完成，保持 batch 16 / 默认线程）

### [Extract] MangaOCR GPU 子进程用于 overlap 模式（>20 页自动开启）
- 说明：串行 30 页实测 MANGA_OCR_GPU_WORKER=1：OCR 101.5s → 16.0s，提取 118s → 61.5s，LLM 未被挤出（翻译 81s 与之前相同），全程 175.9s（与 overlap CPU 版 173.2s 持平）。overlap 模式也启动它：提取 GPU OCR + 同时翻译，预期 30 页约 130-140s
- 涉及文件/模块：main.py（overlap 分支在流水线前调用 start_manga_gpu_worker；提取结束 unload_ocr_models → stop_manga_gpu_worker）
- 实现要点：默认 auto = 超过 20 页（MANGA_OCR_GPU_MIN_PAGES）才启用，短任务子进程启动开销不划算；MANGA_OCR_GPU_WORKER=1/0 强制开/关；GPU 不可用或子进程启动失败自动回退 CPU
- 实测（第5巻 1-30 页 overlap，修复 B33 后）：173.2s → 144.5s（x1.2），提取 129.9s → 55.5s，翻译 87s 与之前相同（GPU OCR 和 LLM 同时推理没有互相拖慢），LLM 未被挤出；瓶颈转为翻译（translate_done 118.5s）
- 相关测试：tests/test_manga_gpu_worker.py（`test_overlap_starts_worker_and_unload_stops_it`、页数门槛）
- 状态：Done

### [Translate] 专有名词库按漫画分开（通用于不同漫画）
- 说明：删除全局 proper_nouns.json / proper_nouns_auto.json 和 local_translator 里写死的《図書館の大魔術師》人名表；每部漫画一个独立词库，名字不会串到别的漫画
- 涉及文件/模块：core/engine.py（`series_key_from_filename`、`set_active_series`、`_save_auto_terms`）、core/local_translator.py（改用当前漫画的词库）、main.py（每个任务开始时切换词库；上传接口可选 `series` 字段覆盖）、data/terms/
- 实现要点：
  - 系列名 = PDF 文件名去掉卷号/话数/括号标签/末尾数字（「図書館の大魔術師 第5巻」→「図書館の大魔術師」，「One_Piece_Vol.12」→「One Piece」），所以同一系列各卷共用；文件名不分大小写
  - data/terms/<系列>.json = 人工词库（第一次翻译时自动建空文件 `{}`，可手动填「原文: 译名」）；data/terms/<系列>.auto.json = LLM 自动发现（只做提示，不强制替换；AUTO_PROPER_NOUNS=1 才开启，gitignored）
  - 原有 295 条 + local_translator 的 13 条英文名已迁移到 data/terms/図書館の大魔術師.json（共 308 条），自动词条迁到 .auto.json
  - glossary.json（通用 IT/AI 术语）保持全局
  - 任务是一个接一个执行的，所以切换全局词库是安全的
- 相关测试：tests/test_terms.py（系列名解析、不同漫画互不影响、自动词条只写入当前漫画、local_translator 不再有写死的名字）
- 状态：Done

### 任务结束后自动清理缓存与输出文件（static / cache / uploads / 临时渲染目录）
- 说明：不再在磁盘上累积文件。页缓存 data/cache/<pdf-hash>/ 和临时渲染目录 pdf_render_* 在任务结束时（成功、失败或取消）删除；static/ 里的输出文件（PDF / ZIP / DOC / JSON）下载一次后立即删除，没下载的在下一个任务开始时删除；如果下载后突然关机或服务崩溃，下次启动时清空 static/、data/cache/、data/uploads/ 和残留的 pdf_render_* 目录
- 涉及文件/模块：main.py（`clear_dir_contents`、`clear_leftovers`（startup 调用）、worker `finally`、下载接口 `BackgroundTask(_remove_quietly)`）、core/pipeline.py（C# 入口同样在 `finally` 删除页缓存和临时目录）、templates/index.html（`oneShotDownload`：按钮点一次后变灰，因为文件已被删除）
- 实现要点：
  - 取代 B10 的「static 保留最新 20 个文件」策略；之前任务的下载链接不再保留
  - 页缓存只在任务内有效，同一 PDF 再次翻译会重新翻译所有页（用户确认）
  - 下载接口因为会删除文件，新增路径校验：只允许 STATIC_DIR 下一层的普通文件，拒绝 `..\xxx`、`..`、`.gitkeep`（返回 400）
  - .gitkeep 占位文件保留
- 相关测试：tests/test_low_bugs.py（`test_clear_dir_contents_removes_files_and_dirs_but_keeps_gitkeep`、`test_cleanup_wiring`、`test_download_rejects_path_escape`）；TestClient 实测：第一次下载 200 且文件被删，第二次 404，`..%5Cmain.py` 400
- 状态：Done

### [Home] Excel 校对台本 + 校对回填（取代 Word / 隐藏 JSON 与调节面板）
- 说明：翻译完成后提供「Excel 校对台本 (.xlsx)」（取代 Word .doc）。用户在 Excel 里修改「译文」一栏，再在页面切换到「✏️ 校对回填」，上传同一个 PDF + 改好的 Excel，系统不调用 LLM，直接把修改后的译文重新消字、排版进 PDF。页面上隐藏了「实时重绘调优」面板和 JSON 下载按钮（固定使用默认参数）；选择 PDF 时不再自动打开调节面板或跑预览
- 涉及文件/模块：core/document_skill.py（`generate_bilingual_xlsx`、`read_corrections`、`match_corrections`）、main.py（上传接口可选 `corrections` 字段、`_load_corrections`、worker 回填分支）、templates/index.html（模式切换、Excel 上传区、按钮改名）、requirements.txt（openpyxl>=3.1）
- 实现要点：
  - 「台本」工作表：页码 / 气泡 / 原文 / 译文（可修改）+ 隐藏的 bbox 四栏；隐藏的 meta 工作表：PDF 的 MD5、源语言、重绘参数
  - 回填时重新提取气泡，按「同页 + bbox IoU ≥ 0.5」匹配 Excel 行（id 只用于平局），所以 id 有变化也能对上；没匹配到的气泡保持原图
  - 校验：只接受 .xlsx、≤20MB、必须有「台本」和 meta 工作表且表头未改；PDF 的 MD5 必须与台本一致，否则返回 400 并提示「PDF 与 Excel 台本不匹配」；上传的 xlsx 解析后立即删除
  - 回填强制使用串行模式，页码与参数沿用台本里记录的值；跳过 LLM 启动和重启
  - 译文单元格强制为文本类型，以「=」开头的译文不会变成公式
  - main.py 不再写页缓存（每个任务结束都会删除，不可能被复用）；这也修复了真实测试里缓存目录缺失导致任务失败的问题
- 相关测试：tests/test_excel_reinsert.py（导出→修改→读回、公式安全、id 变化 / bbox 偏移匹配、坏文件和外来文件拒绝、PDF 不匹配 400 且不留文件），已接入 test_pre_commit；真实测试：第5巻 第10页 5 个气泡，Excel 回填 5/5 匹配，45.9s 完成，未启动 LLM
- 状态：Done

### [Home] 移除「LLM 推理监控与性能看板」
- 说明：右侧卡片不再显示标题与三个指标（翻译吞吐速率 Tokens/Sec、投机草稿接受率、首字捕捉延迟）；下方 SSE 控制台保留
- 涉及文件/模块：templates/index.html（删除看板 HTML、`.metrics-grid`/`.metric-*` CSS、`metricSpeed/Rate/Time` 相关 JS）
- 实现要点：后端 SSE 仍可能携带 `data.metrics`，前端直接忽略，无需改后端
- 相关测试：tests/test_low_bugs.py `test_llm_metrics_board_removed_from_page`
- 状态：Done
