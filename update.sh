#!/bin/bash
# ======================================================
#  繁體中文字幕生成器 - 快速更新腳本
#  只更新 Python 套件，不重建 venv，不影響 Whisper 模型
# ======================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "======================================"
echo "  繁體中文字幕生成器 - 更新套件"
echo "======================================"

if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "❌ 尚未安裝，請先執行 install.sh"
    exit 1
fi

echo "📦 更新 openai-whisper 與 zhconv..."
"$SCRIPT_DIR/venv/bin/pip" install --upgrade pip --quiet
"$SCRIPT_DIR/venv/bin/pip" install --upgrade openai-whisper zhconv

echo ""
echo "✅ 套件更新完成！"
echo "   Whisper 模型快取（~/.cache/whisper/）完整保留。"
echo "   重新啟動 start.command 即可使用。"
