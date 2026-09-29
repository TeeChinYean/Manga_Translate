# A/B the RAM of llama-server under RAM-saving flags (same model/args as llm_models.json).
# STOP the normal LLM first (it holds the VRAM), then run with the web app's Python:
#   python scratch\llama_ram_ab.py
# Each variant: start on port 18090 -> wait healthy -> send one ~2k-token request (fills the
# prompt cache like a real translate batch) -> read Private / Working set -> stop.
import json, os, subprocess, sys, time, urllib.request

RAG = r"C:\Users\Work\Desktop\project\qwen_turbovec_rag"
cfg = json.load(open(os.path.join(RAG, "llm_models.json"), encoding="utf-8"))
model = cfg["models"][0]
PORT = 18090
BASE = [cfg["llama_exe"], "-m", model["path"], "--fit", "on", "--fit-target", str(cfg.get("vram_reserve_mb", 204)),
        "--fit-ctx", str(model.get("min_ctx", 8192)), "-dev", cfg.get("device", "Vulkan1"), "-np", "1",
        "-ctk", cfg["cache_type_k"], "-ctv", cfg["cache_type_v"], "-fa", "on", "-b", "2048",
        "-ub", str(model.get("ub", 512)), "--reasoning", "off", "--jinja", "--port", str(PORT), "--host", "127.0.0.1"]
VARIANTS = [
    ("baseline (now)", []),
    ("--cache-ram 0", ["--cache-ram", "0"]),
    ("--no-mmap", ["--no-mmap"]),
    ("--cache-ram 0 --no-mmap", ["--cache-ram", "0", "--no-mmap"]),
    ("+ --ctx-checkpoints 1", ["--cache-ram", "0", "--no-mmap", "--ctx-checkpoints", "1"]),
]


def mem(pid):
    ps = ("$p=Get-Process -Id %d; '{0} {1}' -f [int]($p.PrivateMemorySize64/1MB),[int]($p.WorkingSet64/1MB)" % pid)
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout.split()
    return (int(out[0]), int(out[1])) if len(out) == 2 else (None, None)


def healthy(timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200:
                return True
        except Exception:
            time.sleep(1)
    return False


def one_request():
    text = "吾輩は猫である。名前はまだ無い。" * 120        # ~2k tokens of Japanese
    body = json.dumps({"messages": [{"role": "user", "content": "翻译成中文：" + text}], "max_tokens": 64}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t = time.time()
    urllib.request.urlopen(req, timeout=180).read()
    return time.time() - t


def vram():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "?"


print(f"{'variant':28s} {'load s':>6s} {'req s':>6s} {'Private MB':>10s} {'WS MB':>7s} {'VRAM MB':>8s}")
for name, extra in VARIANTS:
    t0 = time.time()
    p = subprocess.Popen(BASE + extra, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not healthy():
            print(f"{name:28s} failed to start"); continue
        load = time.time() - t0
        req = one_request()
        time.sleep(2)
        priv, ws = mem(p.pid)
        print(f"{name:28s} {load:6.1f} {req:6.1f} {priv:>10} {ws:>7} {vram():>8}", flush=True)
    finally:
        p.kill(); p.wait(); time.sleep(3)
