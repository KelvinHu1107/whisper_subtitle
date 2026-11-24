#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
繁體中文字幕生成器 v2.0
使用 OpenAI Whisper 轉錄影片語音，輸出可匯入 Final Cut Pro 的 SRT 檔案

更新記錄:
  v3.3 - 修正 SRT 編碼（移除 BOM）、加強 min_gap（200ms）、FCPXML 移除 adjust-transform
  v3.2 - 換用 Basic Title 模板（零 keypoint，完全靜態無動畫，修正文字不可見問題）
  v3.1 - 修正 FCPXML DTD 子元素順序錯誤（adjust-transform 移至 text-style-def 之後）
  v3.0 - FCPXML 輸出（每行字幕為 FCP Title Clip，可個別改字體／顏色）、字幕時間偏移調整
  v2.2 - 幻覺偵測過濾、防幻覺解碼參數、重複段落清理
  v2.0 - 雙進度條即時顯示、詳細 Log 檔、自動更新支援
  v1.0 - 初始版本
"""

VERSION = "3.3"

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import sys
import subprocess
import logging
import traceback
import queue
import re
import time
from pathlib import Path
from datetime import datetime


# ── macOS PATH 補丁 ───────────────────────────────────────────────────────────
# 用 .app 點開時不繼承 shell PATH，Homebrew 安裝的 ffmpeg / ffprobe 會找不到。
# 在這裡主動把常見 Homebrew 路徑加入 PATH，確保 Whisper 和 ffprobe 都能找到。
for _brew_bin in ("/opt/homebrew/bin", "/usr/local/bin"):
    if os.path.isdir(_brew_bin) and _brew_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _brew_bin + ":" + os.environ.get("PATH", "")


# ── 取消例外（BaseException，穿透 Whisper 內部的 except Exception）───────────
class _CancelledError(BaseException):
    """由 WhisperCapture.write() 拋出，中斷阻塞中的 transcribe() 呼叫。"""
    pass

# ── 常數 ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR    = Path(__file__).parent
LOG_DIR       = SCRIPT_DIR / "logs"
WHISPER_CACHE = Path.home() / ".cache" / "whisper"

SUPPORTED_EXTENSIONS = [
    ".mp4", ".mkv", ".avi", ".mov", ".wmv",
    ".flv", ".webm", ".m4v", ".ts", ".mts", ".m2ts", ".mpg", ".mpeg"
]

MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]

MODEL_HINTS = {
    "tiny":     "極快，精確度低（測試用）",
    "base":     "快速，基本精確度",
    "small":    "速度與精確度均衡",
    "medium":   "精確，中文性價比最高",
    "large-v2": "⭐ 中文最推薦：高精確度、幻覺少",
    "large-v3": "最新模型，精確度最高，但中文偶有幻覺（已啟用過濾）",
}

MODEL_FILES = {
    "tiny":     "tiny.pt",
    "base":     "base.pt",
    "small":    "small.pt",
    "medium":   "medium.pt",
    "large-v2": "large-v2.pt",
    "large-v3": "large-v3.pt",
}

# ── 辨識品質預設 ──────────────────────────────────────────────────────────────
# no_speech_threshold：越低越不容易跳過開頭/輕聲段落（預設 0.6，此處調低）
# word_timestamps 對中文反而降低精準度（DTW 字元對齊噪音大），保持關閉
QUALITY_PRESETS = {
    # ⚠️  temperature 不設定 → Whisper 使用預設 tuple (0.0,0.2,0.4,0.6,0.8,1.0)
    #     保留「溫度回退機制」：卡住時自動升溫跳脫，避免無限重複同一句話
    #     condition_on_previous_text=False：每段獨立解碼，防止錯誤向後傳播
    "快速": dict(
        beam_size                   = 5,
        no_speech_threshold         = 0.45,  # 降低（預設 0.6）：避免跳過開頭輕聲/音樂片段
        compression_ratio_threshold = 2.4,
        logprob_threshold           = -1.0,
        condition_on_previous_text  = False, # 🔑 防止重複迴圈
    ),
    "精準": dict(
        beam_size                   = 5,
        no_speech_threshold         = 0.35,  # 降低（預設 0.6）：更保守，確保不遺漏開頭
        compression_ratio_threshold = 2.2,
        logprob_threshold           = -1.0,
        condition_on_previous_text  = False, # 🔑 防止重複迴圈
        # ⚠️  word_timestamps=True 對中文段落時間軸反而有負面影響，不啟用
    ),
}

# ── 內容提示詞預設 ─────────────────────────────────────────────────────────────
# initial_prompt 讓 Whisper 知道語境，大幅減少同音字錯誤
PROMPT_PRESETS = {
    # ⚠️  提示詞應為「關鍵字」而非完整句子。
    #     完整句子會被 Whisper 當成「之前的對話」並不斷複製輸出。
    "一般繁體中文（預設）":  "繁體中文",
    "台灣口語":              "台灣 口語 繁體中文",
    "旅遊 / 生活 vlog":     "旅遊 景點 美食 交通 台灣 繁體中文",
    "會議 / 訪談":           "會議 訪談 討論 繁體中文",
    "新聞 / 報導":           "新聞 報導 繁體中文",
    "（不使用提示詞）":       "",
}

# ── Log 設定 ──────────────────────────────────────────────────────────────────

def setup_file_logger() -> tuple:
    """建立 Log 檔（logs/ 目錄），回傳 (logger, log_path)"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"subtitle_{ts}.log"

    logger = logging.getLogger("whisper_subtitle")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)

    logger.info(f"繁體中文字幕生成器 v{VERSION} 啟動")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    logger.info(f"Script dir: {SCRIPT_DIR}")
    logger.info(f"Whisper cache: {WHISPER_CACHE}")

    return logger, log_path


# ── 繁體中文轉換 ──────────────────────────────────────────────────────────────

_opencc_cvt = None          # opencc 實例（第一次使用時初始化）
_tc_lib: str = ""           # 實際使用的函式庫名稱（供 log 顯示）

def _convert_to_traditional(text: str) -> str:
    """
    簡體 → 繁體中文轉換。
    優先使用 opencc（最精確），其次 zhconv，兩者皆無則回傳原文。
    使用 module-level 快取避免重複建立 OpenCC 物件。
    """
    global _opencc_cvt, _tc_lib

    # ── 1. opencc-python-reimplemented ────────────────────────────────
    if _tc_lib in ("opencc", ""):
        try:
            if _opencc_cvt is None:
                import opencc as _oc
                # s2twp = Simplified → Traditional (Taiwan) 含片語轉換
                _opencc_cvt = _oc.OpenCC("s2twp")
                _tc_lib = "opencc"
            return _opencc_cvt.convert(text)
        except Exception:
            _opencc_cvt = None
            _tc_lib = "zhconv"   # 降級

    # ── 2. zhconv ─────────────────────────────────────────────────────
    if _tc_lib in ("zhconv", ""):
        try:
            import zhconv as _zc
            result = _zc.convert(text, "zh-tw")
            _tc_lib = "zhconv"
            return result
        except Exception:
            _tc_lib = "none"

    # ── 3. 無可用函式庫 ────────────────────────────────────────────────
    return text


def _tc_lib_name() -> str:
    """回傳目前使用的繁體轉換函式庫名稱（用於啟動 log）。"""
    # 嘗試一次，觸發上面的初始化邏輯
    _convert_to_traditional("简体")
    return _tc_lib or "none"


# ── SRT / FCPXML 工具 ─────────────────────────────────────────────────────────

def fmt_srt_time(seconds: float) -> str:
    ms      = int(round(seconds % 1 * 1000))
    total_s = int(seconds)
    s       = total_s % 60
    m       = (total_s // 60) % 60
    h       = total_s // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _prepare_segments(segments: list,
                       delay_ms: int = 0,
                       min_gap_ms: int = 200) -> list:
    """
    深度複製 segments，依序套用：
    1. 時間偏移（delay_ms > 0 = 延後，< 0 = 提前；修正 Whisper 系統性偏早問題）
    2. 依 start 排序（Whisper 偶爾輸出亂序片段）
    3. 確保每段 end > start（排除零長度或倒序段落）
    4. 相鄰字幕最小間距 min_gap_ms（預設 200ms = 5 frames@25fps）
       ── 對 SRT/FCPXML 皆適用，防止播放器或 FCP 字幕匯入器出錯
    """
    segs = [dict(s) for s in segments]

    # 1. 時間偏移
    if delay_ms != 0:
        offset = delay_ms / 1000.0
        for s in segs:
            s["start"] = max(0.0, s["start"] + offset)
            s["end"]   = max(s["start"] + 0.001, s["end"] + offset)

    # 2. 依 start 排序（確保間距修正可以線性處理）
    segs.sort(key=lambda s: s["start"])

    # 3. 確保 end > start
    for s in segs:
        if s["end"] <= s["start"]:
            s["end"] = s["start"] + 0.04   # 補足最少 1 frame@25fps

    # 4. 相鄰最小間距（縮短前一段的 end，不移動後一段的 start）
    min_gap = min_gap_ms / 1000.0
    for i in range(len(segs) - 1):
        gap = segs[i + 1]["start"] - segs[i]["end"]
        if gap < min_gap:
            new_end = segs[i + 1]["start"] - min_gap
            # 只縮短，不讓 end < start（避免把段落壓垮）
            segs[i]["end"] = max(segs[i]["start"] + 0.04, new_end)

    return segs


def build_srt(segments: list, to_tc: bool,
              min_gap_ms: int = 200, delay_ms: int = 0) -> str:
    """
    產生 SRT 字串。
    min_gap_ms：相鄰字幕的最小間距（毫秒，預設 200ms）。
    delay_ms  ：時間偏移（正值 = 延後，負值 = 提前；修正 Whisper 系統性偏早問題）。
    """
    segs = _prepare_segments(segments, delay_ms=delay_ms, min_gap_ms=min_gap_ms)

    idx    = 1
    blocks = []
    for seg in segs:
        text = seg["text"].strip()
        if not text:
            continue
        if to_tc:
            text = _convert_to_traditional(text)
        start = fmt_srt_time(seg["start"])
        end   = fmt_srt_time(seg["end"])
        blocks.append(f"{idx}\n{start} --> {end}\n{text}")
        idx += 1
    # 每筆之間以空行分隔，最後一筆後也加空行（符合 SRT 規範，FCP 字幕匯入需要）
    return "\n\n".join(blocks) + "\n\n" if blocks else ""


def build_fcpxml(segments: list, to_tc: bool, duration: float,
                 delay_ms: int = 0, video_name: str = "Subtitles") -> str:
    """
    產生 Final Cut Pro FCPXML（版本 1.11）。

    每行字幕轉為 lane 1 上的獨立 Basic Title Clip（PETemplates/Bumper:Opener）：
      - 零 keypoint 動畫（完全靜態，直接切入切出，文字不會淡入/縮放/移動）
      - 位置使用模板預設值（不設 adjust-transform；匯入後可在 FCP Video Inspector 調整）
      - 匯入 FCP 後，每個字幕出現在 Timeline 的 Connected Storyline
      - 可在 Text Inspector 個別更改字體、大小、顏色、位置
      - 不需要任何第三方套件（使用 Python 內建 xml.etree.ElementTree）

    delay_ms：與 build_srt 相同的時間偏移，確保 SRT 與 FCPXML 時間軸一致。
    """
    import xml.etree.ElementTree as ET

    def _fcp_t(seconds: float) -> str:
        """
        秒數 → FCPXML frame-aligned rational time（25fps）。
        FCP 要求所有 offset/duration 必須是 frameDuration(100/2500s=1/25s) 的整數倍，
        否則匯入時報 'not on an edit frame boundary' 錯誤。
        """
        frames = max(0, round(seconds * 25))   # 四捨五入到最近的 frame
        return f"{frames * 100}/2500s"         # 與 frameDuration 同分母

    segs = _prepare_segments(segments, delay_ms=delay_ms, min_gap_ms=80)
    segs = [s for s in segs if s.get("text", "").strip()]
    if not segs:
        return ""

    total_dur = max(duration, segs[-1]["end"] if segs else 1.0)

    # ── 根節點 ────────────────────────────────────────────
    root = ET.Element("fcpxml", {"version": "1.11"})

    # ── Resources ────────────────────────────────────────
    resources = ET.SubElement(root, "resources")
    ET.SubElement(resources, "format", {
        "id":            "r1",
        "name":          "FFVideoFormat1080p25",
        "frameDuration": "100/2500s",
        "width":         "1920",
        "height":        "1080",
        "colorSpace":    "1-1-1 (Rec. 709)",
    })
    ET.SubElement(resources, "effect", {
        "id":   "r2",
        "name": "Basic Title",
        # PETemplates/Bumper:Opener/Basic Title：
        #   ✅ 零 keypoint 動畫（透明度/縮放/位置全部靜態，字幕直接切入切出）
        #   ✅ 零 timemarker（無 Build In / Build Out 效果）
        #   ✅ 1 個文字圖層（layer name="Main"）
        # Expand/Gravity/Reveal 雖然 timemarkerset 為空，但仍有 Blending(opacity)
        # 與 Scale keypoint 動畫，導致短字幕片段在渲染時幾乎不可見。
        "uid":  ".../Titles.localized/Bumper:Opener.localized/Basic Title.localized/Basic Title.moti",
    })

    # ── Library → Event → Project → Sequence ─────────────
    library  = ET.SubElement(root,    "library")
    event    = ET.SubElement(library, "event",   {"name": "WhisperSubtitle 字幕"})
    project  = ET.SubElement(event,   "project",  {"name": video_name})
    sequence = ET.SubElement(project, "sequence", {
        "format":      "r1",
        "duration":    _fcp_t(total_dur),
        "tcStart":     "0s",
        "tcFormat":    "NDF",
        "audioLayout": "stereo",
        "audioRate":   "48k",
    })
    spine = ET.SubElement(sequence, "spine")

    # ── 主 Gap（放置所有 Title 的容器）─────────────────────
    gap = ET.SubElement(spine, "gap", {
        "name":     "Gap",
        "offset":   "0s",
        "duration": _fcp_t(total_dur),
        "start":    "0s",
    })

    # ── 每行字幕 → Basic Title Clip（零動畫，Y=-300 像素）──────────
    for idx, seg in enumerate(segs, start=1):
        text = seg["text"].strip()
        if to_tc:
            text = _convert_to_traditional(text)

        start = seg["start"]
        dur   = max(1/25, seg["end"] - start)   # 最少 1 frame @ 25fps = 0.04s
        label = (text[:20] + "…") if len(text) > 20 else text
        ts_id = f"ts{idx}"

        title = ET.SubElement(gap, "title", {
            "name":     label,
            "lane":     "1",
            "offset":   _fcp_t(start),
            "ref":      "r2",
            "duration": _fcp_t(dur),
            "start":    "0s",
        })

        # ── DTD 規定 title 子元素順序：param* → text* → text-style-def* ──
        # 文字內容
        text_elem = ET.SubElement(title, "text")
        ts_ref    = ET.SubElement(text_elem, "text-style", {"ref": ts_id})
        ts_ref.text = text

        # 文字樣式（FCP 匯入後可自由覆蓋）
        ts_def = ET.SubElement(title, "text-style-def", {"id": ts_id})
        ET.SubElement(ts_def, "text-style", {
            "font":      "Helvetica Neue",
            "fontSize":  "52",
            "fontFace":  "Regular",
            "fontColor": "1 1 1 1",     # 白色不透明
            "alignment": "center",
        })

    # ── 輸出格式化 XML ────────────────────────────────────
    ET.indent(root, space="    ")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE fcpxml>\n'
        + ET.tostring(root, encoding="unicode")
    )


# ── 幻覺過濾 ──────────────────────────────────────────────────────────────────

# Whisper 已知幻覺字串（訓練資料污染，常見於靜音或雜訊片段）
_HALLUCINATION_RE = re.compile(
    r'Amara\.org'                                              # 最常見
    r'|字幕.*(?:Amara|社群提供|翻譯提供)'
    r'|(?:Chinese|英文|中文|繁體|简体).{0,15}(?:字幕|subtitle).{0,20}(?:提供|by|翻譯|製作)'
    r'|subtitle[s]?\s+(?:provided\s+)?by\s+\w'               # 英文版
    r'|MBC\s*뉴스'                                             # 韓文新聞幻覺
    r'|NHK|TBS|BBC\s+News'                                    # 廣播幻覺
    r'|Thank\s+you\s+for\s+watching'
    r'|請訂閱.*頻道'
    r'|歡迎訂閱',
    re.IGNORECASE,
)


def filter_segments(segments: list) -> tuple:
    """
    過濾幻覺段落與連續重複段落。
    回傳 (filtered_list, removed_count, removed_details)
    """
    kept    = []
    removed = []
    prev_text = None
    repeat_run = 0

    for seg in segments:
        text = seg.get("text", "").strip()

        # 1. 空白段落
        if not text:
            removed.append(("空白", text, seg.get("start", 0)))
            continue

        # 2. 已知幻覺字串
        if _HALLUCINATION_RE.search(text):
            removed.append(("幻覺字串", text, seg.get("start", 0)))
            continue

        # 3. 連續重複（同一句話連出現 3 次以上視為幻覺迴圈）
        if text == prev_text:
            repeat_run += 1
            if repeat_run >= 2:          # 已重複第 3 次，丟棄
                removed.append(("重複迴圈", text, seg.get("start", 0)))
                continue
        else:
            repeat_run = 0
            prev_text  = text

        kept.append(seg)

    return kept, len(removed), removed


# ── 模型下載（支援續傳與自動重試）────────────────────────────────────────────

def download_whisper_model(
    model_name:   str,
    cache_dir:    Path,
    *,
    progress_cb,   # (pct: float) -> None
    log_cb,        # (msg: str, tag: str = "") -> None
    status_cb,     # (msg: str) -> None
    cancel_check,  # () -> bool  ── True 表示使用者要求取消
    max_retries:  int = 10,
) -> bool:
    """
    下載 Whisper 模型，具備：
    ・HTTP Range 續傳（斷網恢復後從中斷處繼續）
    ・SHA256 完整性驗證
    ・自動重試（最多 max_retries 次，逐步拉長等待）
    ・即時進度與速度顯示

    回傳 True = 成功，False = 取消或失敗。
    """
    import whisper
    import hashlib
    import urllib.request
    import urllib.error

    # 從 whisper 內部取得下載 URL
    model_url = whisper._MODELS.get(model_name)
    if not model_url:
        log_cb(f"❌ 找不到模型 {model_name!r} 的下載 URL", "err")
        return False

    expected_sha256 = model_url.split("/")[-2]   # SHA256 嵌在 URL 倒數第二段
    dest = cache_dir / f"{model_name}.pt"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 重試等待秒數（5→10→20→30→60→60→…）
    retry_delays = [5, 10, 20, 30, 60, 60, 60, 60, 60, 60]

    def _fmt_size(n: int) -> str:
        return f"{n/1024**3:.2f} GB" if n >= 1024**3 else f"{n/1024**2:.0f} MB"

    for attempt in range(max_retries):
        if cancel_check():
            return False

        existing = dest.stat().st_size if dest.exists() else 0

        try:
            # ── 取得完整檔案大小 ──────────────────────────────
            head = urllib.request.Request(model_url, method="HEAD",
                                          headers={"User-Agent": "whisper-subtitle/2.2"})
            with urllib.request.urlopen(head, timeout=15) as r:
                total_size = int(r.headers.get("Content-Length", 0))

            # 若檔案已存在且大小相符，直接驗 SHA256
            if existing == total_size > 0:
                log_cb("   🔍 驗證現有檔案完整性…", "dim")
                with open(dest, "rb") as f:
                    if hashlib.sha256(f.read()).hexdigest() == expected_sha256:
                        log_cb("   ✅ 檔案完整，跳過下載", "ok")
                        progress_cb(100)
                        return True
                log_cb("   ⚠️  檔案損壞，重新下載", "warn")
                dest.unlink()
                existing = 0

            # ── 設定 Range 請求（續傳）───────────────────────
            resuming = 0 < existing < total_size
            headers  = {"User-Agent": "whisper-subtitle/2.2"}
            if resuming:
                headers["Range"] = f"bytes={existing}-"
                log_cb(f"   ↩️  從 {_fmt_size(existing)} 處續傳"
                       f"（剩餘 {_fmt_size(total_size - existing)}）", "dim")
            else:
                log_cb(f"   ⬇️  開始下載（總大小 {_fmt_size(total_size)}）", "dim")

            req = urllib.request.Request(model_url, headers=headers)

            with urllib.request.urlopen(req, timeout=60) as resp:
                # 若伺服器不支援 Range（回 200），從頭開始
                if resuming and resp.status == 200:
                    log_cb("   ⚠️  伺服器不支援續傳，從頭開始", "warn")
                    resuming  = False
                    existing  = 0

                content_length = int(resp.headers.get("Content-Length", 0))
                full_total = (existing + content_length) if resuming else content_length
                mode       = "ab" if resuming else "wb"
                downloaded = existing
                t0 = t_chunk = time.time()
                bytes_since = 0

                with open(dest, mode) as f:
                    while True:
                        if cancel_check():
                            log_cb("   ⚠️  已取消下載", "warn")
                            return False

                        chunk = resp.read(65536)   # 64 KB
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded   += len(chunk)
                        bytes_since  += len(chunk)

                        # 每 0.5 秒更新一次進度 + 速度
                        now = time.time()
                        if now - t_chunk >= 0.5:
                            speed_mb = bytes_since / (now - t_chunk) / 1024**2
                            pct = downloaded / full_total * 100 if full_total else 0
                            progress_cb(min(98, pct))
                            status_cb(
                                f"下載 {model_name}  "
                                f"{_fmt_size(downloaded)} / {_fmt_size(full_total)}"
                                f"  ({pct:.0f}%)  {speed_mb:.1f} MB/s"
                            )
                            t_chunk    = now
                            bytes_since = 0

            # ── SHA256 驗證 ───────────────────────────────────
            log_cb("   🔍 驗證 SHA256…", "dim")
            progress_cb(99)
            with open(dest, "rb") as f:
                got_hash = hashlib.sha256(f.read()).hexdigest()

            if got_hash != expected_sha256:
                log_cb(f"   ❌ SHA256 不符，檔案可能損壞，刪除重試", "err")
                dest.unlink()
                existing = 0
                continue   # 重新嘗試

            progress_cb(100)
            elapsed = time.time() - t0
            avg_speed = (downloaded - existing) / elapsed / 1024**2 if elapsed > 0 else 0
            log_cb(f"   ✅ 下載完成！({elapsed:.0f}s，平均 {avg_speed:.1f} MB/s)", "ok")
            return True

        except (urllib.error.URLError, ConnectionError,
                TimeoutError, OSError) as exc:
            saved = dest.stat().st_size if dest.exists() else 0
            delay = retry_delays[min(attempt, len(retry_delays) - 1)]
            log_cb(f"   ⚠️  網路中斷：{exc}", "warn")
            log_cb(f"   💾 已保存 {_fmt_size(saved)}，{delay} 秒後自動重試"
                   f"（第 {attempt + 1}/{max_retries} 次）", "warn")

            # 倒數等待，每秒更新狀態列
            for remaining in range(delay, 0, -1):
                if cancel_check():
                    return False
                status_cb(f"⚠️  網路中斷，{remaining} 秒後重試…")
                time.sleep(1)

    log_cb("   ❌ 超過最大重試次數，下載失敗", "err")
    return False


# ── 影片時長 ──────────────────────────────────────────────────────────────────

def get_video_duration(path: str) -> float:
    """用 ffprobe 取得影片秒數，失敗回傳 0"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             path],
            capture_output=True, text=True, timeout=30
        )
        val = result.stdout.strip()
        return float(val) if val else 0.0
    except Exception:
        return 0.0


# ── Whisper 輸出攔截 ──────────────────────────────────────────────────────────

class WhisperCapture:
    """
    Context manager：攔截 whisper verbose=True 的 stdout/stderr 輸出，
    解析段落時間戳以計算進度。

    支援格式：
      [MM:SS.mmm --> MM:SS.mmm]  文字
      [HH:MM:SS.mmm --> HH:MM:SS.mmm]  文字

    當 cancel_check() 返回 True 時，write() 拋出 _CancelledError（BaseException），
    從 Whisper 內部中斷阻塞中的 transcribe() 呼叫。
    """

    # 同時支援 MM:SS.mmm 和 HH:MM:SS.mmm
    _RE = re.compile(
        r'\[(?:(\d+):)?(\d+):(\d+\.\d+)\s*-->\s*[^\]]+\]\s*(.*)'
    )

    def __init__(self, duration: float, progress_cb, segment_cb,
                 cancel_check=None):
        """
        duration     : 影片秒數（用於計算百分比）
        progress_cb  : (pct: int) -> None   ，pct 為 0-95
        segment_cb   : (line: str, text: str) -> None
        cancel_check : () -> bool  ── 返回 True 時立即拋出 _CancelledError
        """
        self.duration     = max(duration, 1.0)
        self.progress_cb  = progress_cb
        self.segment_cb   = segment_cb
        self.cancel_check = cancel_check
        self._buf         = ""
        self._cancelled   = False
        self._real_out    = sys.__stdout__
        self._real_err    = sys.__stderr__

    def __enter__(self):
        self._real_out = sys.stdout
        self._real_err = sys.stderr
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._real_out
        sys.stderr = self._real_err
        # 只在未取消時才處理殘餘 buffer
        if self._buf.strip() and not self._cancelled:
            self._parse(self._buf)
        self._buf = ""
        # 不壓制例外，讓 _CancelledError 繼續傳播
        return False

    def write(self, text: str):
        # 🔑 取消偵測：Whisper 每輸出一個段落就會呼叫 write()
        #    BaseException 可穿透 Whisper 內部的 except Exception 區塊
        if self.cancel_check and self.cancel_check():
            self._cancelled = True
            raise _CancelledError("用戶已取消辨識")

        # 同時保留到真實 stdout（方便 debug）
        try:
            self._real_out.write(text)
        except Exception:
            pass
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._parse(line)

    def flush(self):
        try:
            self._real_out.flush()
        except Exception:
            pass

    def _parse(self, line: str):
        m = self._RE.match(line.strip())
        if not m:
            return
        h    = float(m.group(1) or 0)
        mins = float(m.group(2))
        secs = float(m.group(3))
        text = m.group(4).strip()

        current_sec = h * 3600 + mins * 60 + secs
        pct = min(95, int(current_sec / self.duration * 100))
        self.progress_cb(pct)

        if text:
            self.segment_cb(line.strip(), text)


# ── 主程式 ────────────────────────────────────────────────────────────────────

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title(f"繁體中文字幕生成器  v{VERSION}")
        self.geometry("1020x760")
        self.resizable(True, True)
        self.minsize(860, 660)

        # 狀態
        self.video_files: list = []
        self.output_dir         = tk.StringVar(value=str(Path.home() / "Downloads"))
        self.model_var          = tk.StringVar(value="large-v2")
        self.convert_tc         = tk.BooleanVar(value=True)
        self.quality_var        = tk.StringVar(value="精準")
        self.prompt_preset_var  = tk.StringVar(value="一般繁體中文（預設）")
        self.prompt_var         = tk.StringVar(
                                    value=PROMPT_PRESETS["一般繁體中文（預設）"])
        self.subtitle_delay_ms  = tk.IntVar(value=0)
        self.export_fcpxml      = tk.BooleanVar(value=False)
        self.is_running         = False
        self._cancel            = False
        self._log_q: queue.Queue = queue.Queue()
        self._cached_model      = None
        self._cached_model_name = None
        self._animate_loading   = False
        self._caffeinate_proc   = None   # 防閒置休眠的子程序

        # Log 檔
        self._logger, self._log_path = setup_file_logger()

        self._build_ui()
        self._poll_log()
        self.protocol("WM_DELETE_WINDOW", self._on_window_close)

        # 啟動訊息
        self._log(f"✅ 繁體中文字幕生成器 v{VERSION} 就緒", "ok")
        self._log(f"   Log 檔：{self._log_path}", "dim")
        # 偵測繁體轉換函式庫
        tc_lib = _tc_lib_name()
        if tc_lib == "opencc":
            self._log("🔤 繁體轉換：opencc（最精確）", "ok")
        elif tc_lib == "zhconv":
            self._log("🔤 繁體轉換：zhconv（建議安裝 opencc-python-reimplemented 以提升精確度）", "dim")
        else:
            self._log("⚠️  繁體轉換：找不到 opencc 或 zhconv，輸出可能包含簡體字！", "warn")
            self._log("   請執行 install.sh 重新安裝依賴", "warn")
        self._check_model_cache()

    # ── 模型快取偵測 ──────────────────────────────────────────────────────────

    def _is_model_cached(self, name: str) -> bool:
        fname = MODEL_FILES.get(name, "")
        return bool(fname) and (WHISPER_CACHE / fname).exists()

    def _check_model_cache(self):
        downloaded = [n for n in MODELS if self._is_model_cached(n)]
        if downloaded:
            self._log(f"📦 已下載的模型：{', '.join(downloaded)}", "dim")
        else:
            self._log("📦 尚未下載任何模型（首次使用時會自動下載）", "dim")

    def _model_hint_text(self, name: str) -> str:
        hint   = MODEL_HINTS.get(name, "")
        cached = "✅ 已下載" if self._is_model_cached(name) else "⬇ 首次使用需下載"
        return f"{hint}　　{cached}"

    # ── UI 建構 ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── 頂部標題（固定）────────────────────────────────
        header = tk.Frame(self, pady=12, padx=18)
        header.pack(side="top", fill="x")
        # 結束按鈕（先 pack 到右側，才能正確佔位）
        ttk.Button(header, text="結束並清除 Log",
                   command=self._quit_app).pack(side="right")
        tk.Label(header, text="繁體中文字幕生成器",
                 font=("SF Pro Display", 22, "bold")).pack(side="left")
        tk.Label(header, text=f"  ·  OpenAI Whisper  ·  v{VERSION}",
                 font=("SF Pro Text", 12), fg="#888").pack(side="left", pady=5)
        ttk.Separator(self, orient="horizontal").pack(side="top", fill="x")

        # ── 底部進度列（固定，必須在 expand=True 的主面板之前宣告）──
        ttk.Separator(self, orient="horizontal").pack(side="bottom", fill="x")
        self._build_bottom()

        # ── 主面板（中間自由擴展，最後宣告）────────────────
        pw = tk.PanedWindow(self, orient="horizontal", sashwidth=5,
                            sashrelief="flat", bd=0)
        pw.pack(side="top", fill="both", expand=True, padx=10, pady=8)

        left  = tk.Frame(pw)
        right = tk.Frame(pw)
        pw.add(left,  width=490, minsize=300)
        pw.add(right, minsize=290)

        self._build_file_panel(left)
        self._build_settings_panel(right)

    def _build_file_panel(self, parent):
        row = tk.Frame(parent)
        row.pack(fill="x", pady=(0, 6))
        tk.Label(row, text="影片檔案",
                 font=("SF Pro Text", 13, "bold")).pack(side="left")
        btns = tk.Frame(row)
        btns.pack(side="right")
        ttk.Button(btns, text="新增檔案",
                   command=self._add_files).pack(side="left", padx=(0, 4))
        ttk.Button(btns, text="移除選取",
                   command=self._remove_sel).pack(side="left", padx=(0, 4))
        ttk.Button(btns, text="清除全部",
                   command=self._clear_files).pack(side="left")

        lf = tk.Frame(parent, relief="sunken", bd=1)
        lf.pack(fill="both", expand=True)
        sb = ttk.Scrollbar(lf)
        sb.pack(side="right", fill="y")
        self.listbox = tk.Listbox(lf, selectmode="extended",
                                  font=("Menlo", 11),
                                  yscrollcommand=sb.set,
                                  activestyle="none")
        self.listbox.pack(fill="both", expand=True, padx=2, pady=2)
        sb.config(command=self.listbox.yview)
        self.listbox.bind("<Delete>",    lambda _: self._remove_sel())
        self.listbox.bind("<BackSpace>", lambda _: self._remove_sel())

        self.file_lbl = tk.Label(parent, text="尚未選擇任何檔案",
                                 fg="#888", font=("SF Pro Text", 11))
        self.file_lbl.pack(anchor="w", pady=(5, 0))

    def _build_settings_panel(self, parent):
        tk.Label(parent, text="設定",
                 font=("SF Pro Text", 13, "bold")).pack(anchor="w")

        sf = ttk.LabelFrame(parent, padding=(10, 8))
        sf.pack(fill="x", pady=(4, 0))
        sf.columnconfigure(1, weight=1)

        # 模型選擇
        tk.Label(sf, text="Whisper 模型").grid(row=0, column=0, sticky="w", pady=5)
        cb = ttk.Combobox(sf, textvariable=self.model_var,
                          values=MODELS, state="readonly", width=14)
        cb.grid(row=0, column=1, sticky="w", padx=(10, 0), pady=5)
        cb.bind("<<ComboboxSelected>>", self._on_model_change)

        self.hint_lbl = tk.Label(sf, text=self._model_hint_text(self.model_var.get()),
                                 fg="#888", font=("SF Pro Text", 10))
        self.hint_lbl.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=(0, 4))

        # 輸出目錄
        tk.Label(sf, text="輸出目錄").grid(row=2, column=0, sticky="w", pady=5)
        od = tk.Frame(sf)
        od.grid(row=2, column=1, sticky="ew", padx=(10, 0), pady=5)
        tk.Entry(od, textvariable=self.output_dir,
                 font=("Menlo", 10)).pack(side="left", fill="x", expand=True)
        ttk.Button(od, text="瀏覽",
                   command=self._pick_output).pack(side="right", padx=(4, 0))

        # 繁體中文轉換
        tk.Label(sf, text="繁體中文").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Checkbutton(sf, text="自動轉換為繁體中文",
                        variable=self.convert_tc
                        ).grid(row=3, column=1, sticky="w", padx=(10, 0), pady=5)

        # 辨識品質
        tk.Label(sf, text="辨識品質").grid(row=4, column=0, sticky="w", pady=5)
        qf = tk.Frame(sf)
        qf.grid(row=4, column=1, sticky="w", padx=(10, 0), pady=5)
        ttk.Radiobutton(qf, text="快速", value="快速",
                        variable=self.quality_var,
                        command=self._on_quality_change).pack(side="left")
        ttk.Radiobutton(qf, text="精準（建議）", value="精準",
                        variable=self.quality_var,
                        command=self._on_quality_change).pack(side="left", padx=(12, 0))
        self.quality_hint = tk.Label(sf, text=self._quality_hint_text("精準"),
                                     fg="#888", font=("SF Pro Text", 9))
        self.quality_hint.grid(row=5, column=1, sticky="w", padx=(10, 0), pady=(0, 4))

        # 內容提示詞（最影響同音字準確度）
        ttk.Separator(sf, orient="horizontal").grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=6)

        tk.Label(sf, text="內容提示",
                 font=("SF Pro Text", 10, "bold")).grid(
            row=7, column=0, sticky="nw", pady=(2, 0))
        tk.Label(sf, text="← 最有效改善同音字錯誤",
                 fg="#16a34a", font=("SF Pro Text", 9)).grid(
            row=8, column=0, sticky="w", padx=(0, 4))

        # 預設選單
        prompt_cb = ttk.Combobox(
            sf, textvariable=self.prompt_preset_var,
            values=list(PROMPT_PRESETS.keys()),
            state="readonly", width=20)
        prompt_cb.grid(row=7, column=1, sticky="w", padx=(10, 0), pady=(2, 4))
        prompt_cb.bind("<<ComboboxSelected>>", self._on_prompt_preset_change)

        # 自訂輸入框
        self.prompt_entry = tk.Entry(sf, textvariable=self.prompt_var,
                                     font=("SF Pro Text", 10))
        self.prompt_entry.grid(row=8, column=1, sticky="ew", padx=(10, 0), pady=(0, 6))
        tk.Label(sf,
                 text="⚠️ 請填關鍵字（如「旅遊 青島 美食」），勿填完整句子",
                 fg="#d97706", font=("SF Pro Text", 9)).grid(
            row=9, column=1, sticky="w", padx=(10, 0), pady=(0, 4))

        # ── 字幕時間偏移 ──────────────────────────────────────────────────────
        ttk.Separator(sf, orient="horizontal").grid(
            row=10, column=0, columnspan=2, sticky="ew", pady=6)

        tk.Label(sf, text="時間校正",
                 font=("SF Pro Text", 10, "bold")).grid(
            row=11, column=0, sticky="nw", pady=(2, 0))

        delay_row = tk.Frame(sf)
        delay_row.grid(row=11, column=1, sticky="w", padx=(10, 0), pady=(2, 0))
        tk.Label(delay_row, text="字幕延後").pack(side="left")
        ttk.Spinbox(delay_row,
                    textvariable=self.subtitle_delay_ms,
                    from_=-2000, to=2000, increment=50,
                    width=7).pack(side="left", padx=(6, 4))
        tk.Label(delay_row, text="毫秒").pack(side="left")

        tk.Label(sf,
                 text="預設 0。若字幕比聲音早，輸入 +300 ～ +500",
                 fg="#6b7280", font=("SF Pro Text", 9)).grid(
            row=12, column=1, sticky="w", padx=(10, 0), pady=(0, 6))

        # ── FCPXML 輸出 ───────────────────────────────────────────────────────
        ttk.Separator(sf, orient="horizontal").grid(
            row=13, column=0, columnspan=2, sticky="ew", pady=6)

        tk.Label(sf, text="FCPXML",
                 font=("SF Pro Text", 10, "bold")).grid(
            row=14, column=0, sticky="nw", pady=(2, 0))
        ttk.Checkbutton(sf,
                        text="同時輸出 .fcpxml（Final Cut Pro）",
                        variable=self.export_fcpxml).grid(
            row=14, column=1, sticky="w", padx=(10, 0), pady=(2, 0))

        tk.Label(sf,
                 text="每行字幕為獨立 Title Clip，可在 FCP 個別改字體／顏色／位置",
                 fg="#6b7280", font=("SF Pro Text", 9)).grid(
            row=15, column=1, sticky="w", padx=(10, 0), pady=(0, 6))

        # 執行記錄
        log_hdr = tk.Frame(parent)
        log_hdr.pack(fill="x", pady=(14, 5))
        tk.Label(log_hdr, text="執行記錄",
                 font=("SF Pro Text", 13, "bold")).pack(side="left")
        ttk.Button(log_hdr, text="開啟 Log 檔",
                   command=self._open_log).pack(side="right")

        log_f = tk.Frame(parent, relief="sunken", bd=1)
        log_f.pack(fill="both", expand=True)
        lsb = ttk.Scrollbar(log_f)
        lsb.pack(side="right", fill="y")
        self.log_box = tk.Text(log_f, state="disabled", font=("Menlo", 10),
                               wrap="word", yscrollcommand=lsb.set)
        self.log_box.pack(fill="both", expand=True, padx=4, pady=4)
        lsb.config(command=self.log_box.yview)

        self.log_box.tag_config("ok",      foreground="#16a34a")
        self.log_box.tag_config("err",     foreground="#dc2626")
        self.log_box.tag_config("warn",    foreground="#d97706")
        self.log_box.tag_config("dim",     foreground="#6b7280")
        self.log_box.tag_config("segment", foreground="#7c3aed")

    def _build_bottom(self):
        bar = tk.Frame(self, pady=8, padx=16)
        bar.pack(side="bottom", fill="x")

        # ── 整體進度 ──
        row1 = tk.Frame(bar)
        row1.pack(fill="x", pady=(0, 3))
        tk.Label(row1, text="整體進度", font=("SF Pro Text", 10),
                 fg="#555", width=8, anchor="w").pack(side="left")
        self.prog_overall = ttk.Progressbar(row1, mode="determinate", maximum=100)
        self.prog_overall.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.lbl_overall = tk.Label(row1, text="  0%",
                                    font=("Menlo", 10), fg="#555", width=5)
        self.lbl_overall.pack(side="left")

        # ── 當前檔案進度 ──
        row2 = tk.Frame(bar)
        row2.pack(fill="x", pady=(0, 6))
        tk.Label(row2, text="當前檔案", font=("SF Pro Text", 10),
                 fg="#555", width=8, anchor="w").pack(side="left")
        self.prog_file = ttk.Progressbar(row2, mode="determinate", maximum=100)
        self.prog_file.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.lbl_file = tk.Label(row2, text="  0%",
                                 font=("Menlo", 10), fg="#555", width=5)
        self.lbl_file.pack(side="left")

        # ── 狀態 + 按鈕 ──
        row3 = tk.Frame(bar)
        row3.pack(fill="x")

        self.status_lbl = tk.Label(row3, text="準備就緒", fg="#666",
                                   font=("SF Pro Text", 11))
        self.status_lbl.pack(side="left")

        ttk.Button(row3, text="開啟輸出目錄",
                   command=self._open_output).pack(side="right", padx=(6, 0))

        self.cancel_btn = ttk.Button(row3, text="取消",
                                     command=self._request_cancel,
                                     state="disabled")
        self.cancel_btn.pack(side="right", padx=(6, 0))

        self.start_btn = ttk.Button(row3, text="▶  開始生成字幕",
                                    command=self._start)
        self.start_btn.pack(side="right")

    # ── 事件 ──────────────────────────────────────────────────────────────────

    def _quality_hint_text(self, q: str) -> str:
        p = QUALITY_PRESETS.get(q, {})
        return (f"beam_size={p.get('beam_size')}  "
                f"no_speech={p.get('no_speech_threshold')}  "
                f"condition_on_prev=False  temperature=自動回退")

    def _on_quality_change(self):
        self.quality_hint.config(
            text=self._quality_hint_text(self.quality_var.get()))

    def _on_prompt_preset_change(self, _=None):
        key  = self.prompt_preset_var.get()
        text = PROMPT_PRESETS.get(key, "")
        self.prompt_var.set(text)

    def _on_model_change(self, _=None):
        self.hint_lbl.config(text=self._model_hint_text(self.model_var.get()))

    def _add_files(self):
        ext_glob = " ".join(f"*{e}" for e in SUPPORTED_EXTENSIONS)
        paths = filedialog.askopenfilenames(
            title="選擇影片檔案",
            filetypes=[
                ("影片檔案", ext_glob),
                ("MP4",           "*.mp4"),
                ("MKV",           "*.mkv"),
                ("AVI",           "*.avi"),
                ("QuickTime MOV", "*.mov"),
                ("WMV",           "*.wmv"),
                ("WebM",          "*.webm"),
                ("M4V",           "*.m4v"),
                ("TS/MTS/M2TS",   "*.ts *.mts *.m2ts"),
                ("MPEG",          "*.mpg *.mpeg"),
                ("所有檔案",       "*.*"),
            ],
        )
        for p in paths:
            if p not in self.video_files:
                self.video_files.append(p)
                self.listbox.insert("end", Path(p).name)
        self._refresh_label()

    def _remove_sel(self):
        for idx in reversed(self.listbox.curselection()):
            self.video_files.pop(idx)
            self.listbox.delete(idx)
        self._refresh_label()

    def _clear_files(self):
        self.video_files.clear()
        self.listbox.delete(0, "end")
        self._refresh_label()

    def _refresh_label(self):
        n = len(self.video_files)
        self.file_lbl.config(
            text="尚未選擇任何檔案" if n == 0
            else f"共 {n} 個檔案（選取後可按 Delete 移除）"
        )

    def _pick_output(self):
        d = filedialog.askdirectory(title="選擇 SRT 輸出目錄")
        if d:
            self.output_dir.set(d)

    def _open_output(self):
        d = self.output_dir.get()
        if os.path.isdir(d):
            subprocess.run(["open", d])

    def _open_log(self):
        if self._log_path.exists():
            subprocess.run(["open", str(self._log_path)])
        else:
            messagebox.showinfo("提示", f"Log 檔不存在：{self._log_path}")

    def _request_cancel(self):
        self._cancel = True
        self.cancel_btn.config(state="disabled")   # 防止重複點擊
        self._log("⚠️  已收到取消指令，辨識將在當前音訊段落完成後中斷（最多約 30 秒）", "warn")
        self._set_status("⏹  取消中，請稍候…")
        self._logger.warning("使用者請求取消")

    # ── 結束程式 ──────────────────────────────────────────────────────────────

    def _on_window_close(self):
        """視窗 X 按鈕：若辨識中詢問確認，否則直接關閉（Log 保留）"""
        if self.is_running:
            if not messagebox.askyesno("確認關閉", "正在辨識中，確定要強制關閉嗎？\n\nLog 檔將保留。"):
                return
            self._cancel = True
        self._caffeinate_stop()
        self.destroy()

    def _quit_app(self):
        """結束按鈕：關閉程式並刪除本次 Log 檔（含外部檢視器）"""
        # 辨識中需確認
        if self.is_running:
            if not messagebox.askyesno(
                "確認結束",
                "正在辨識中。\n確定要強制結束並刪除 Log 嗎？",
                icon="warning",
            ):
                return
            self._cancel = True

        # ── 1. 關閉 Python 的 FileHandler（釋放檔案描述符）
        for handler in self._logger.handlers[:]:
            try:
                handler.close()
                self._logger.removeHandler(handler)
            except Exception:
                pass

        # ── 2. 關閉外部 Log 檢視器（如 TextEdit、Console）
        self._close_log_viewer()

        # ── 3. 刪除 Log 檔
        try:
            if self._log_path.exists():
                self._log_path.unlink()
        except Exception:
            pass

        # ── 4. 若 logs/ 目錄已空則刪除
        try:
            if LOG_DIR.exists() and not any(LOG_DIR.iterdir()):
                LOG_DIR.rmdir()
        except Exception:
            pass

        # ── 5. 結束
        self._caffeinate_stop()
        self.destroy()

    def _close_log_viewer(self):
        """
        用 lsof 找出開啟 Log 檔的外部程式（排除自身）並送 SIGTERM 關閉。
        例：TextEdit、Console、CotEditor 等。
        """
        if not self._log_path.exists():
            return
        try:
            result = subprocess.run(
                ["lsof", "-t", str(self._log_path)],
                capture_output=True, text=True, timeout=3,
            )
            my_pid = os.getpid()
            for pid_str in result.stdout.strip().splitlines():
                pid_str = pid_str.strip()
                if not pid_str.isdigit():
                    continue
                pid = int(pid_str)
                if pid == my_pid:
                    continue   # 不殺自己
                try:
                    subprocess.run(
                        ["kill", "-TERM", str(pid)],
                        capture_output=True, timeout=2,
                    )
                except Exception:
                    pass
        except Exception:
            pass

    # ── 休眠防護 ──────────────────────────────────────────────────────────────

    def _caffeinate_start(self):
        """啟動 caffeinate -i，防止系統閒置休眠（闔蓋休眠不受影響）"""
        try:
            self._caffeinate_proc = subprocess.Popen(
                ["caffeinate", "-i", "-w", str(os.getpid())],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._log("☕ 防閒置休眠已啟動（闔蓋仍可休眠）", "dim")
            self._logger.info("caffeinate 啟動")
        except Exception as e:
            self._logger.warning(f"caffeinate 啟動失敗：{e}")

    def _caffeinate_stop(self):
        """停止 caffeinate"""
        if self._caffeinate_proc:
            try:
                self._caffeinate_proc.terminate()
            except Exception:
                pass
            self._caffeinate_proc = None
            self._log("☕ 防閒置休眠已停止", "dim")
            self._logger.info("caffeinate 停止")

    def _sleep_watcher(self):
        """
        背景執行緒：偵測系統休眠喚醒（透過時間跳躍）。
        macOS 闔蓋時 process 被暫停，打開後繼續執行，
        此時 time.time() 會出現 > 10 秒的跳躍。
        """
        last = time.monotonic()
        while self.is_running:
            time.sleep(4)
            now   = time.monotonic()
            drift = now - last - 4          # 理想值 ≈ 0
            last  = now
            if drift > 8:                   # 休眠超過 8 秒才提示
                secs = int(drift)
                mins = secs // 60
                display = f"{mins} 分 {secs%60} 秒" if mins else f"{secs} 秒"
                self._log(f"\n⏰ 系統喚醒（休眠約 {display}），繼續辨識中…", "warn")
                self._logger.warning(f"系統喚醒，休眠約 {secs}s")

    # ── 進度控制 ──────────────────────────────────────────────────────────────

    def _set_overall(self, val: float):
        v = max(0.0, min(100.0, val))
        self.after(0, lambda: [
            self.prog_overall.config(value=v),
            self.lbl_overall.config(text=f"{int(v):3d}%")
        ])

    def _set_file(self, val: float):
        v = max(0.0, min(100.0, val))
        self.after(0, lambda: [
            self.prog_file.config(value=v),
            self.lbl_file.config(text=f"{int(v):3d}%")
        ])

    def _set_indeterminate(self, on: bool):
        """載入模型時使用不定進度動畫"""
        def _do():
            if on:
                self.prog_file.config(mode="indeterminate")
                self.prog_file.start(40)
                self.lbl_file.config(text=" ···")
            else:
                self.prog_file.stop()
                self.prog_file.config(mode="determinate", value=0)
                self.lbl_file.config(text="  0%")
        self.after(0, _do)

    def _reset_progress(self):
        self._set_overall(0)
        self._set_file(0)

    # ── 轉錄核心 ──────────────────────────────────────────────────────────────

    def _start(self):
        if self.is_running:
            return
        if not self.video_files:
            messagebox.showwarning("提示", "請先新增影片檔案")
            return
        out = self.output_dir.get()
        if not os.path.isdir(out):
            messagebox.showwarning("提示", "輸出目錄不存在，請重新選擇")
            return

        self.is_running = True
        self._cancel    = False
        self.start_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self._reset_progress()
        self._logger.info("=" * 60)
        self._logger.info(f"開始處理 {len(self.video_files)} 個檔案")

        self._caffeinate_start()
        threading.Thread(target=self._worker,       daemon=True).start()
        threading.Thread(target=self._sleep_watcher, daemon=True).start()

    def _worker(self):
        try:
            self._transcribe_all()
        except Exception as exc:
            tb = traceback.format_exc()
            self._log(f"❌ 未預期錯誤：{exc}", "err")
            self._log(tb, "err")
            self._logger.error(f"未預期錯誤：{exc}\n{tb}")
        finally:
            self.is_running = False
            self._caffeinate_stop()
            self._set_indeterminate(False)
            self.after(0, lambda: [
                self.start_btn.config(state="normal"),
                self.cancel_btn.config(state="disabled"),
            ])

    def _transcribe_all(self):
        # ── 1. Import whisper ─────────────────────────────────────────────────
        try:
            import whisper
        except ImportError:
            self._log("❌ 找不到 whisper，請先執行 install.sh", "err")
            self._set_status("安裝錯誤")
            return

        # ── 2. 載入模型 ───────────────────────────────────────────────────────
        model_name = self.model_var.get()

        if self._cached_model_name != model_name or self._cached_model is None:
            cached = self._is_model_cached(model_name)
            self._logger.info(f"載入模型: {model_name}（已快取: {cached}）")

            # ── 若模型尚未下載，先自行下載（支援續傳）──────────
            if not cached:
                self._log(f"📥 需要下載模型 {model_name}（首次使用）", "warn")
                self._set_status(f"準備下載 {model_name}…")

                ok = download_whisper_model(
                    model_name, WHISPER_CACHE,
                    progress_cb  = self._set_file,
                    log_cb       = self._log,
                    status_cb    = self._set_status,
                    cancel_check = lambda: self._cancel,
                )
                if not ok:
                    if not self._cancel:
                        self._log("❌ 模型下載失敗，請檢查網路後重試", "err")
                        self._set_status("下載失敗")
                    return
                self._set_file(0)
                self.after(0, self._on_model_change)   # 刷新「✅ 已下載」

            # ── 載入模型到記憶體 ─────────────────────────────────
            self._log(f"📥 載入模型 {model_name} 到記憶體…", "dim")
            self._set_status(f"載入模型 {model_name}…")
            self._set_indeterminate(True)

            try:
                self._cached_model      = whisper.load_model(model_name)
                self._cached_model_name = model_name
                self._set_indeterminate(False)
                self._log(f"✅ 模型 {model_name} 就緒", "ok")
                self._logger.info(f"模型 {model_name} 載入完成")
                self.after(0, self._on_model_change)
            except Exception as e:
                tb = traceback.format_exc()
                self._set_indeterminate(False)
                self._log(f"❌ 模型載入失敗：{e}", "err")
                self._log(tb, "err")
                self._logger.error(f"模型載入失敗：{e}\n{tb}")
                self._set_status("模型載入失敗")
                return

        # ── 3. 逐一處理影片 ───────────────────────────────────────────────────
        files   = list(self.video_files)
        total   = len(files)
        out_dir = Path(self.output_dir.get())
        to_tc   = self.convert_tc.get()
        ok_cnt  = 0

        for i, path in enumerate(files):
            if self._cancel:
                self._log("⚠️  已取消", "warn")
                self._logger.warning("使用者取消")
                break

            name = Path(path).name
            stem = Path(path).stem
            # 整體進度的起訖點（每個檔案佔等比份額）
            g_start = (i / total) * 100
            g_end   = ((i + 1) / total) * 100

            self._log(f"\n[{i+1}/{total}] 🎬 {name}")
            self._logger.info(f"[{i+1}/{total}] 開始: {name}")
            self._set_status(f"處理中 {i+1}/{total}：{name}")
            self._set_overall(g_start)
            self._set_file(0)

            try:
                # 取得影片時長
                self._log(f"   🔍 分析影片時長…", "dim")
                self._set_file(2)
                duration = get_video_duration(path)
                if duration > 0:
                    mins = duration / 60
                    self._log(
                        f"   ⏱ 時長：{duration:.1f} 秒（{mins:.1f} 分鐘）", "dim"
                    )
                    self._logger.info(f"   時長: {duration:.1f}s")
                else:
                    self._log("   ⚠️  無法取得時長，進度估算可能不精準", "warn")
                    self._logger.warning("   無法取得時長")

                # 進度回呼
                def make_callbacks(g_s, g_e):
                    def on_progress(pct: int):
                        # pct: 0-95 (WhisperCapture 保證上限 95)
                        file_pct  = 5 + pct * 0.9        # 5% → 90.5%
                        total_pct = g_s + (file_pct / 100) * (g_e - g_s)
                        self._set_file(file_pct)
                        self._set_overall(total_pct)

                    def on_segment(line: str, text: str):
                        # 顯示時套用繁體轉換，方便即時監看
                        tc_text = _convert_to_traditional(text) if to_tc else text
                        ts_end  = line.rfind("]")
                        display = (f"{line[:ts_end+1]}  {tc_text}"
                                   if ts_end >= 0 else line)
                        self._log(f"   {display}", "segment")
                        self._logger.debug(f"   {line}")   # Log 檔保留原始輸出
                        # 狀態列顯示當前辨識內容（截斷）
                        preview = tc_text[:28] + "…" if len(tc_text) > 28 else tc_text
                        self._set_status(f"[{i+1}/{total}] {preview}")

                    return on_progress, on_segment

                on_progress, on_segment = make_callbacks(g_start, g_end)

                # 組合辨識參數
                quality = self.quality_var.get()
                prompt  = self.prompt_var.get().strip()
                params  = dict(QUALITY_PRESETS.get(quality, QUALITY_PRESETS["精準"]))
                if prompt:
                    params["initial_prompt"] = prompt

                self._log(f"   🎙 開始辨識（品質：{quality}）…", "dim")
                self._log(f"   📋 beam_size={params['beam_size']}  "
                          f"temperature=自動回退  "
                          f"no_speech_threshold={params['no_speech_threshold']}", "dim")
                if prompt:
                    self._log(f"   💬 提示詞：{prompt[:50]}{'…' if len(prompt)>50 else ''}", "dim")
                self._logger.info(f"   辨識參數: {params}")
                self._set_file(5)

                # ── 辨識（支援取消中斷）────────────────────────
                result = None
                try:
                    with WhisperCapture(duration, on_progress, on_segment,
                                        cancel_check=lambda: self._cancel):
                        result = self._cached_model.transcribe(
                            path,
                            language="zh",
                            task="transcribe",
                            verbose=True,
                            fp16=False,
                            **params,   # condition_on_previous_text=False 在 params 裡
                        )
                except _CancelledError:
                    self._log(f"   ⚠️  辨識已取消（{name}）", "warn")
                    self._logger.warning(f"   辨識取消: {name}")
                    break

                if result is None:
                    continue

                raw_segs = result.get("segments", [])
                self._log(f"   ✅ 辨識完成，共 {len(raw_segs)} 段（過濾前）", "ok")
                self._logger.info(f"   辨識完成，{len(raw_segs)} 段（過濾前）")
                self._set_file(97)

                # ── 時間軸診斷 ────────────────────────────────
                if raw_segs:
                    first_ts = fmt_srt_time(raw_segs[0]["start"])
                    last_ts  = fmt_srt_time(raw_segs[-1]["end"])
                    self._log(f"   📊 時間軸範圍：{first_ts} → {last_ts}", "dim")
                    # 若第一段不是從頭開始，顯示提示
                    if raw_segs[0]["start"] > 5.0:
                        self._log(
                            f"   ⚠️  注意：字幕從 {first_ts} 開始，前段可能無語音或被 VAD 跳過",
                            "warn",
                        )
                    # 顯示前 3 段時間戳供偵錯
                    self._log("   📋 前 3 段時間戳：", "dim")
                    for j, seg in enumerate(raw_segs[:3]):
                        self._log(
                            f"      [{j+1}] {fmt_srt_time(seg['start'])} → "
                            f"{fmt_srt_time(seg['end'])}  "
                            f"{seg['text'].strip()[:35]}",
                            "dim",
                        )
                    # 異常偵測：end ≤ start 代表時間軸嚴重錯誤
                    bad = [s for s in raw_segs if s["end"] <= s["start"]]
                    if bad:
                        self._log(
                            f"   ❌ {len(bad)} 個段落 end ≤ start，時間軸異常！",
                            "err",
                        )
                        for seg in bad[:3]:
                            self._log(
                                f"      {fmt_srt_time(seg['start'])} → "
                                f"{fmt_srt_time(seg['end'])}: "
                                f"{seg['text'].strip()[:30]}",
                                "err",
                            )

                # 幻覺過濾
                segs, n_removed, removed_details = filter_segments(raw_segs)
                if n_removed > 0:
                    self._log(f"   🧹 過濾 {n_removed} 個幻覺/重複段落：", "warn")
                    for reason, text, start in removed_details:
                        preview = text[:40] + "…" if len(text) > 40 else text
                        self._log(f"      [{reason}] {fmt_srt_time(start)} → {preview!r}", "warn")
                        self._logger.warning(f"   過濾[{reason}] {fmt_srt_time(start)}: {text!r}")
                self._log(f"   📝 保留 {len(segs)} 段字幕", "dim")

                # 取得時間偏移設定
                delay_ms = self.subtitle_delay_ms.get()
                if delay_ms:
                    sign = "+" if delay_ms > 0 else ""
                    self._log(f"   ⏱ 字幕時間偏移：{sign}{delay_ms}ms", "dim")

                # 儲存 SRT
                srt_text = build_srt(segs, to_tc, delay_ms=delay_ms)
                out_path = out_dir / f"{stem}.srt"
                out_path.write_text(srt_text, encoding="utf-8")   # 不加 BOM，FCP 字幕匯入需 UTF-8 純文字
                self._log(f"   💾 已儲存 SRT → {out_path.name}", "ok")
                self._logger.info(f"   儲存 SRT: {out_path}")

                # 儲存 FCPXML（選用）
                if self.export_fcpxml.get():
                    try:
                        fcpxml_str  = build_fcpxml(
                            segs, to_tc, duration,
                            delay_ms=delay_ms,
                            video_name=stem,
                        )
                        fcpxml_path = out_dir / f"{stem}.fcpxml"
                        fcpxml_path.write_text(fcpxml_str, encoding="utf-8")
                        self._log(f"   🎬 已儲存 FCPXML → {fcpxml_path.name}", "ok")
                        self._logger.info(f"   儲存 FCPXML: {fcpxml_path}")
                    except Exception as fcpxml_exc:
                        self._log(f"   ⚠️  FCPXML 輸出失敗：{fcpxml_exc}", "warn")
                        self._logger.warning(f"   FCPXML 失敗: {fcpxml_exc}")

                ok_cnt += 1
                self._set_file(100)
                self._set_overall(g_end)

            except Exception as exc:
                tb = traceback.format_exc()
                self._log(f"   ❌ 失敗：{exc}", "err")
                self._log(f"{tb}", "err")
                self._logger.error(f"   失敗: {exc}\n{tb}")

        # ── 4. 完成 ───────────────────────────────────────────────────────────
        if not self._cancel:
            fcpxml_note = "（含 .fcpxml）" if self.export_fcpxml.get() else ""
            self._log(f"\n🎉 完成！成功 {ok_cnt}/{total} 個檔案{fcpxml_note}", "ok")
            self._log(f"   輸出目錄：{out_dir}", "dim")
            self._log(f"   Log 檔：{self._log_path}", "dim")
            self._logger.info(f"完成！{ok_cnt}/{total} 個檔案{fcpxml_note}")
            self._set_status(f"✅ 完成 {ok_cnt}/{total} 個檔案{fcpxml_note}")
            self._set_overall(100)
            self._set_file(100)

    # ── Log 工具 ──────────────────────────────────────────────────────────────

    def _log(self, msg: str, tag: str = ""):
        self._log_q.put((msg, tag))
        level_map = {"ok": "info", "err": "error",
                     "warn": "warning", "dim": "debug",
                     "segment": "debug"}
        level = level_map.get(tag, "info")
        getattr(self._logger, level)(msg)

    def _poll_log(self):
        try:
            while True:
                msg, tag = self._log_q.get_nowait()
                self.log_box.config(state="normal")
                self.log_box.insert("end", msg + "\n", tag)
                self.log_box.see("end")
                self.log_box.config(state="disabled")
        except queue.Empty:
            pass
        self.after(80, self._poll_log)

    def _set_status(self, msg: str):
        self.after(0, lambda: self.status_lbl.config(text=msg))


# ── 進入點 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
