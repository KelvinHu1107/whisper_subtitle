#!/bin/bash
# ======================================================
#  繁體中文字幕生成器 - 安裝／更新腳本
#  可重複執行：venv 已存在則跳過建立，不影響 Whisper 模型
# ======================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================"
echo "  繁體中文字幕生成器 - 安裝依賴"
echo "======================================"
echo ""

# ── Homebrew ──────────────────────────────────────────
if ! command -v brew &>/dev/null; then
    echo "📦 安裝 Homebrew（需要輸入系統密碼）..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    if [ -f /opt/homebrew/bin/brew ]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
fi
echo "✅ Homebrew: $(brew --version | head -1)"

# ── Python 3.12 ───────────────────────────────────────
PYTHON312="/opt/homebrew/bin/python3.12"
if [ ! -f "$PYTHON312" ]; then
    echo ""
    echo "📦 安裝 Python 3.12..."
    brew install python@3.12
fi
if ! brew list python-tk@3.12 &>/dev/null; then
    echo "📦 安裝 Python tkinter 支援..."
    brew install python-tk@3.12
fi
echo "✅ $($PYTHON312 --version)"

# ── ffmpeg ────────────────────────────────────────────
if ! command -v ffmpeg &>/dev/null; then
    echo ""
    echo "📦 安裝 ffmpeg..."
    brew install ffmpeg
fi
echo "✅ ffmpeg: $(ffmpeg -version 2>&1 | head -1)"

# ── Python venv（已存在則跳過，保留已安裝套件）────────
echo ""
if [ -d "$SCRIPT_DIR/venv" ]; then
    echo "✅ 虛擬環境已存在，跳過建立"
else
    echo "📦 建立 Python 虛擬環境（Python 3.12）..."
    "$PYTHON312" -m venv "$SCRIPT_DIR/venv"
    echo "✅ 虛擬環境建立完成"
fi

# ── 安裝 / 更新 Python 套件 ───────────────────────────
echo ""
echo "📦 安裝／更新 Python 套件（openai-whisper、zhconv、opencc）..."
"$SCRIPT_DIR/venv/bin/pip" install --upgrade pip --quiet
"$SCRIPT_DIR/venv/bin/pip" install --upgrade openai-whisper zhconv

# opencc-python-reimplemented：比 zhconv 更精確的繁體轉換
# 安裝失敗不中止腳本（某些系統可能缺少 C 編譯器）
echo "📦 安裝 opencc（更精確的簡體→繁體轉換）..."
"$SCRIPT_DIR/venv/bin/pip" install --upgrade opencc-python-reimplemented 2>/dev/null \
    && echo "✅ opencc 安裝完成" \
    || echo "⚠️  opencc 安裝失敗（已有 zhconv 作為備援，功能不受影響）"

# ── WhisperSubtitle.app：確保可執行且移除 Gatekeeper 隔離 ──
APP_BUNDLE="$SCRIPT_DIR/WhisperSubtitle.app"
if [ -d "$APP_BUNDLE" ]; then
    chmod +x "$APP_BUNDLE/Contents/MacOS/launcher" 2>/dev/null || true
    xattr -dr com.apple.quarantine "$APP_BUNDLE" 2>/dev/null || true
    echo "✅ WhisperSubtitle.app 已就緒（可直接雙擊啟動）"
fi

echo ""
echo "======================================"
echo "  ✅ 安裝完成！"
echo ""
echo "  📝 Whisper 語言模型快取位於 ~/.cache/whisper/"
echo "     重新安裝不會刪除已下載的模型。"
echo ""
echo "  🚀 啟動方式（擇一）："
echo "     • 雙擊 WhisperSubtitle.app   ← 建議，直接開啟不開終端機"
echo "     • 雙擊 start.command         ← 備用方式"
echo "======================================"
