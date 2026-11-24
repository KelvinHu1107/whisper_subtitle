# whisper_subtitle

A Traditional Chinese subtitle generator and a browser-based subtitle editor, built around OpenAI Whisper and FFmpeg. macOS only (macOS 12.0 or later).

## Screenshots

![Subtitle generator](docs/screenshot-1.png)

![Web editor](docs/screenshot-2.png)

## Subtitle generator

The desktop app transcribes video or audio with Whisper and converts the recognized text to Traditional Chinese (OpenCC `s2twp`, with `zhconv` as a fallback).

After installation, launch it by double-clicking `WhisperSubtitle.app`, or by double-clicking `start.command`.

## Web editor

The editor is a local Flask app (`web_editor.py`) that opens at `http://localhost:5500`. Launch it by double-clicking `SubtitleEditor.app`, or by running `./start_web_editor.command`.

- Transcribe a video or audio file with Whisper (`tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`), with “fast” and “accurate” quality presets
- Convert recognized text to Traditional Chinese
- Filter common Whisper hallucinations (credit titles, “please subscribe”, repeated trailers, and similar lines)
- Edit timings on a zoomable timeline with a waveform, drag and resize, magnetic snapping, and long-press to add a cue
- Drag a subtitle in two dimensions on the video overlay, with snap guides
- Style panel for font, size, bold, italic, color, and background
- Auto-align timings after transcription
- Session save / load / clear, and a close-project flow
- Export SRT, VTT, or ASS
- Burn subtitles into the video with FFmpeg, including HEVC 10-bit sources and videos that have no audio track, with a visible FFmpeg error log if burn-in fails

## Setup

```bash
./install.sh
```

This installs or reuses Homebrew, Python 3.12 (plus `python-tk@3.12`), FFmpeg, and a project `venv` with `openai-whisper` and `zhconv`. It also tries to install `opencc-python-reimplemented` and continues with `zhconv` only if that fails. Already-downloaded Whisper weights in `~/.cache/whisper/` are left in place.

To update Python packages later without rebuilding the venv:

```bash
./update.sh
```

## Stack

- Python 3.12 · Flask · openai-whisper · FFmpeg
- OpenCC (`s2twp`) and/or zhconv for Simplified → Traditional Chinese
- macOS `.app` launchers (minimum system version 12.0)
