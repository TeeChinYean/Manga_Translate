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
) -> None:
    """
    Run producer -> translator -> renderer concurrently with bounded queues.

    - `source`: async iterator of pages (the producer).
    - `translate_page(page)`: returns the page to render, or None to skip rendering
      (e.g. page restored from cache).
    - `render_page(page)`: renders one page.

    Raises the first stage exception, or PipelineCancelled if cancelled.
    """
    abort = asyncio.Event()
    errors: list = []
    q_extracted: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
    q_translated: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
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

    async def translator():
        try:
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

    async def renderer():
        try:
            while True:
                page = await get(q_translated)
                if page is _DONE or check_cancel():
                    break
                await render_page(page)
        except Exception as ex:
            fail(ex)

    await asyncio.gather(producer(), translator(), renderer())

    if errors:
        raise errors[0]
    if state["cancelled"]:
        raise PipelineCancelled("Task cancelled by user.")
