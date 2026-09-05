#!/bin/bash
# 雙擊此檔案即可在 macOS Finder 中直接啟動
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="$SCRIPT_DIR/venv/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "❌ 尚未安裝依賴，請先在終端機執行："
    echo "   cd ~/whisper_subtitle && bash install.sh"
    echo ""
    read -p "按 Enter 關閉..." _
    exit 1
fi

# ── 啟動 Python 應用程式 ──────────────────────────────
# nohup：Terminal 關閉後應用程式繼續執行
# 啟動錯誤暫存到 startup.log 供偵錯
mkdir -p "$SCRIPT_DIR/logs"
STARTUP_LOG="$SCRIPT_DIR/logs/startup.log"

nohup "$PYTHON" "$SCRIPT_DIR/app.py" > "$STARTUP_LOG" 2>&1 &
PYTHON_PID=$!

# ── 等待 Python 初始化（tkinter 視窗需約 1 秒）────────
sleep 1

# ── 確認程式是否正常啟動 ─────────────────────────────
if kill -0 "$PYTHON_PID" 2>/dev/null; then
    # 啟動成功：清除啟動 log
    rm -f "$STARTUP_LOG"

    # 關鍵：把這個 Terminal 視窗設成「shell 結束後自動關閉，不詢問」
    # closing behavior 1 = always close（不跳確認對話框）
    osascript -e \
        'tell application "Terminal" to set closing behavior of front tab of front window to 1' \
        2>/dev/null || true

    # shell 自然結束 → Terminal 偵測到 shell 退出 → 自動關閉視窗
else
    # 啟動失敗：顯示錯誤訊息，等待使用者讀完
    echo ""
    echo "❌ 程式啟動失敗！錯誤訊息如下："
    echo "─────────────────────────────────"
    cat "$STARTUP_LOG" 2>/dev/null || echo "（無法讀取錯誤訊息）"
    echo "─────────────────────────────────"
    echo ""
    echo "請執行 install.sh 重新安裝，或回報上述錯誤。"
    echo ""
    read -p "按 Enter 關閉..." _
fi
