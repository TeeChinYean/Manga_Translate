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
- 状态：Done（待用户实测对比结果）
