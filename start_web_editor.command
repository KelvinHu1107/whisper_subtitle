#!/bin/bash
# 字幕編輯器 Web 版 - 雙擊此檔案啟動
cd "$(dirname "$0")"

# 檢查 venv
if [ ! -f "venv/bin/python" ]; then
  echo "找不到 venv，請先執行 install.sh"
  read -p "按 Enter 結束"
  exit 1
fi

# 安裝 flask（如果尚未安裝）
./venv/bin/pip install flask -q 2>/dev/null

echo "====================================="
echo "  字幕編輯器 Web 版 啟動中…"
echo "  瀏覽器將自動開啟"
echo "  關閉此視窗即可停止伺服器"
echo "====================================="

./venv/bin/python web_editor.py
