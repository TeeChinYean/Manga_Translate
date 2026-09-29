#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deadlock-safe 3-stage async pipeline runner (extract -> translate -> render).

Why this exists (BUG.md B1):
    The previous inline implementation used bounded queues with plain
    `await queue.put(...)`. When a downstream stage died, the upstream stage
    blocked forever on a full queue, `asyncio.gather` never returned, and the
    single global task worker was stuck until the server restarted.

Guarantees:
    * Any stage error sets a shared abort event; every blocking put/get races
      against that event, so no stage can wait forever.
    * The first error is re-raised to the caller after all stages exit.
    * Cooperative cancellation via `is_cancelled()` stops all stages cleanly.
    * Dependency-free (stdlib asyncio only) so it can be unit tested anywhere.
"""

import asyncio
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

_DONE = object()


class PipelineCancelled(Exception):
    """Raised when `is_cancelled()` became true while the pipeline was running."""


async def _race(awaitable: Awaitable, abort: asyncio.Event):
    """Await `awaitable` unless `abort` fires first. Returns (ok, result)."""
    op = asyncio.ensure_future(awaitable)
    stop = asyncio.ensure_future(abort.wait())
    try:
        await asyncio.wait({op, stop}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        stop.cancel()
    if op.done() and not op.cancelled():
        return True, op.result()
    op.cancel()
    return False, None


async def run_three_stage_pipeline(
    source: AsyncIterator[Any],
    translate_page: Callable[[Any], Awaitable[Optional[Any]]],
    render_page: Callable[[Any], Awaitable[None]],
    is_cancelled: Callable[[], bool] = lambda: False,
    queue_size: int = 3,
    hold_render_until_source_done: bool = False,
    on_source_done: Optional[Callable[[], Awaitable[None]]] = None,
    translate_batch: Optional[Callable[[list], Awaitable[list]]] = None,
    batch_lines: int = 0,
    page_lines: Callable[[Any], int] = lambda page: 1,
    render_concurrency: int = 1,
    hold_render_until_translate_done: bool = False,
) -> None:
    """
    Run producer -> translator -> renderer concurrently with bounded queues.

    - `source`: async iterator of pages (the producer).
    - `translate_page(page)`: returns the page to render, or None to skip rendering
      (e.g. page restored from cache).
    - `render_page(page)`: renders one page.

    - `hold_render_until_source_done`: "overlap" mode. Rendering starts only after the
      producer is exhausted and `on_source_done()` (e.g. unload OCR models) has finished;
      translation keeps running the whole time. The translated queue is unbounded in this
      mode so translation never waits for the (not yet started) renderer.

    - `translate_batch(pages) -> pages_to_render` + `batch_lines`: buffer extracted pages until
      they hold >= batch_lines lines (page_lines(page) each) or extraction ends, then translate
      them in ONE call, so the LLM sees dialogue across pages. Used instead of translate_page.

    - `render_concurrency`: number of pages rendered at the same time (renderer workers).

    - `hold_render_until_translate_done`: rendering starts only after ALL translation finished
      (extract + translate still overlap). For GPU LaMa: it must not share the 4 GB card with
      the LLM while the LLM is still translating (B28). The translated queue is unbounded.

    Raises the first stage exception, or PipelineCancelled if cancelled.
    """
    abort = asyncio.Event()
    errors: list = []
    q_extracted: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
    hold_any = hold_render_until_source_done or hold_render_until_translate_done
    q_translated: asyncio.Queue = asyncio.Queue(maxsize=0 if hold_any else queue_size)
    render_gate = asyncio.Event()
    if not hold_render_until_source_done:
        render_gate.set()
    translate_gate = asyncio.Event()
    if not hold_render_until_translate_done:
        translate_gate.set()
    state = {"cancelled": False}

    def fail(exc: BaseException):
        errors.append(exc)
        abort.set()

    def check_cancel() -> bool:
        if is_cancelled():
            state["cancelled"] = True
            abort.set()
        return abort.is_set()

    async def put(q: asyncio.Queue, item) -> bool:
        if abort.is_set():
            return False
        ok, _ = await _race(q.put(item), abort)
        return ok

    async def get(q: asyncio.Queue):
        if abort.is_set():
            return _DONE
        ok, item = await _race(q.get(), abort)
        return item if ok else _DONE

    async def producer():
        try:
            async for page in source:
                if check_cancel():
                    break
                if not await put(q_extracted, page):
                    break
        except Exception as ex:
            fail(ex)
        finally:
            aclose = getattr(source, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    pass
            await put(q_extracted, _DONE)
            if on_source_done is not None and not abort.is_set():
                try:
                    await on_source_done()
                except Exception as ex:
                    fail(ex)
            render_gate.set()

    async def translator():
        try:
            if translate_batch is not None and batch_lines > 0:
                await batched_translator()
                return
            while True:
                page = await get(q_extracted)
                if page is _DONE or check_cancel():
                    break
                out = await translate_page(page)
                if out is None:
                    continue
                if not await put(q_translated, out):
                    break
        except Exception as ex:
            fail(ex)
        finally:
            await put(q_translated, _DONE)
            translate_gate.set()

    async def batched_translator():
        buf, lines = [], 0
        while True:
            page = await get(q_extracted)
            done = page is _DONE
            if check_cancel():
                return
            if not done:
                buf.append(page)
                lines += max(0, int(page_lines(page)))
            if buf and (done or lines >= batch_lines):
                for out in await translate_batch(buf):
                    if not await put(q_translated, out):
                        return
                buf, lines = [], 0
            if done:
                return

    async def renderer():
        try:
            await _race(render_gate.wait(), abort)
            await _race(translate_gate.wait(), abort)
            while True:
                page = await get(q_translated)
                if page is _DONE:
                    # Hand the end marker on to the next renderer worker
                    await put(q_translated, _DONE)
                    break
                if check_cancel():
                    break
                await render_page(page)
        except Exception as ex:
            fail(ex)

    workers = [renderer() for _ in range(max(1, int(render_concurrency)))]
    await asyncio.gather(producer(), translator(), *workers)

    if errors:
        raise errors[0]
    if state["cancelled"]:
        raise PipelineCancelled("Task cancelled by user.")
