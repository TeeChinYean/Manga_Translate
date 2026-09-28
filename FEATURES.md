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
