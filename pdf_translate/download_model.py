import os
import urllib.request
import json
import shutil

# Target directory
target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manga-ocr-base")
if os.path.exists(target_dir):
    print(f"清理旧的下载文件夹: {target_dir}")
    shutil.rmtree(target_dir)
os.makedirs(target_dir, exist_ok=True)

# Files to download from HF mirror
base_url = "https://hf-mirror.com/kha-white/manga-ocr-base/resolve/main/"
files = [
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.txt",
    "special_tokens_map.json",
    "pytorch_model.bin"
]

print("开始从 hf-mirror 直接下载模型文件 (无需任何第三方依赖)...")

for f in files:
    url = base_url + f
    out_path = os.path.join(target_dir, f)
    print(f"正在下载 {f} ...")
    try:
        # User-agent is sometimes required by mirrors
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(out_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print(f"✅ {f} 下载成功!")
    except urllib.error.HTTPError as e:
        if e.code == 404 and f != "pytorch_model.bin" and f != "config.json":
            print(f"⚠️ {f} 不存在 (404), 已跳过 (非必需文件)。")
        else:
            print(f"❌ 下载 {f} 失败: {e}")
            print("请检查网络连通性。")
            exit(1)
    except Exception as e:
        print(f"❌ 下载 {f} 失败: {e}")
        print("请检查网络连通性。")
        exit(1)

print(f"\n🎉 所有模型文件已经成功下载到: {target_dir}")
print("现在您可以直接重启 uvicorn 服务器了！")
