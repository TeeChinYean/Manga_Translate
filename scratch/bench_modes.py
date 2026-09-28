# Benchmark the three pipeline modes against the running server (start_web_app.bat first).
# Usage (repo root):
#   pdf_translate\.venv\Scripts\python scratch\bench_modes.py "C:\path\to\manga.pdf" --pages 1-20
#   add --modes serial,overlap to pick modes; --chunk 36 sets serial-mode lines per LLM call.
import argparse, json, os, time
import httpx

API = "http://127.0.0.1:8000"


def run(pdf, mode, pages, chunk):
    with open(pdf, "rb") as f:
        r = httpx.post(f"{API}/api/v1/translate/upload", timeout=120,
                       files={"file": (os.path.basename(pdf), f, "application/pdf")},
                       data={"source_lang": "Japanese", "page_range": pages, "force_retranslate": "true",
                             "pipeline_mode": mode, "context_chunk_size": str(chunk)})
    task_id = r.json()["task_id"]
    t0, event = time.time(), None
    with httpx.stream("GET", f"{API}/api/v1/translate/status/{task_id}", timeout=None) as s:
        for line in s.iter_lines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and event in ("complete", "error"):
                data = json.loads(line[5:])
                return event, round(time.time() - t0, 1), data
    return "disconnected", round(time.time() - t0, 1), {}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--pages", default="1-20")
    ap.add_argument("--modes", default="stream,overlap,serial")
    ap.add_argument("--chunk", type=int, default=36)
    a = ap.parse_args()
    rows = []
    for mode in a.modes.split(","):
        print(f"== {mode} ...", flush=True)
        ev, wall, data = run(a.pdf, mode, a.pages, a.chunk)
        rows.append((mode, ev, wall, data.get("stage_times"), data.get("warning") or data.get("message", "")))
        print(f"   {ev} in {wall}s  stages={data.get('stage_times')}", flush=True)
    print("\nmode      result    wall(s)  stage_times")
    for mode, ev, wall, st, note in rows:
        print(f"{mode:<9} {ev:<9} {wall:<8} {st} {note}")
    print("\nNote: 'overlap'/'serial' unload OCR at the end of extraction, so the NEXT run reloads it;"
          " run each mode twice or reorder to compare warm vs cold.")
