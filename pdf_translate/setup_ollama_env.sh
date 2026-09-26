#!/bin/bash
# ⚡ RTX 3050 4GB Ollama 终极性能优化配置脚本
# 作用: 将 Flash Attention + KV Cache q8_0 + 并发锁写入 systemd 服务

echo "⚡ 正在写入 Ollama 系统服务优化配置..."

sudo mkdir -p /etc/systemd/system/ollama.service.d

sudo tee /etc/systemd/system/ollama.service.d/override.conf > /dev/null << 'EOF'
[Service]
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
Environment="OLLAMA_NUM_PARALLEL=1"
EOF

echo "✅ 配置写入完成！内容如下："
cat /etc/systemd/system/ollama.service.d/override.conf

echo ""
echo "🔄 正在重启 Ollama 服务使配置生效..."
sudo systemctl daemon-reload
sudo systemctl restart ollama

echo ""
echo "🎉 全部完成！当前 Ollama 优化配置已激活："
echo "   OLLAMA_FLASH_ATTENTION=1  → 闪电注意力，加速长文本推理"
echo "   OLLAMA_KV_CACHE_TYPE=q8_0 → KV Cache 压缩为 8-bit，节省约 50% 显存"
echo "   OLLAMA_NUM_PARALLEL=1     → 单任务模式，防止显存峰值叠加"
