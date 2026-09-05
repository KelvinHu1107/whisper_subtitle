#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
字幕編輯器 - Web 版本 v3.0
啟動：./venv/bin/python web_editor.py  →  http://localhost:5500
"""
import sys, os, re, uuid, threading, subprocess, tempfile, traceback
from pathlib import Path

for _d in ("/opt/homebrew/bin", "/usr/local/bin"):
    if os.path.isdir(_d) and _d not in os.environ.get("PATH",""):
        os.environ["PATH"] = _d + ":" + os.environ.get("PATH","")

try:
    from flask import Flask, request, jsonify, send_file, send_from_directory
except ImportError:
    print("需要 Flask：./venv/bin/pip install flask"); sys.exit(1)

WHISPER_CACHE = Path.home() / ".cache" / "whisper"
UPLOAD_DIR    = Path(tempfile.gettempdir()) / "whisper_web_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MODELS     = ["tiny","base","small","medium","large-v2","large-v3"]
MODEL_FILES= {"tiny":"tiny.pt","base":"base.pt","small":"small.pt",
              "medium":"medium.pt","large-v2":"large-v2.pt","large-v3":"large-v3.pt"}
QUALITY_PRESETS = {
    "快速": dict(beam_size=5,no_speech_threshold=0.45,compression_ratio_threshold=2.4,
                 logprob_threshold=-1.0,condition_on_previous_text=False),
    "精準": dict(beam_size=5,no_speech_threshold=0.35,compression_ratio_threshold=2.2,
                 logprob_threshold=-1.0,condition_on_previous_text=False),
}
_HALL_RE = re.compile(
    r'Amara\.org'
    r'|字幕.*(?:Amara|社群提供|翻譯提供|志願者|義工)'
    r'|中文字幕志願者|字幕志願者'
    r'|subtitle[s]?\s+(?:provided\s+)?by\s+'
    r'|Thank\s+you\s+for\s+watching'
    r'|請[訂订]閱.{0,10}[頻频]道'
    r'|歡迎[訂订]閱'
    r'|謝謝收看|谢谢收看|謝謝觀看|谢谢观看'
    r'|李宗盛'
    r'|字幕製作|字幕翻譯|字幕校對'
    r'|transcribed\s+by|translated\s+by',
    re.IGNORECASE)

_opencc = None
def _to_tc(t):
    global _opencc
    try:
        if _opencc is None:
            import opencc as _oc; _opencc = _oc.OpenCC("s2twp")
        return _opencc.convert(t)
    except Exception:
        try:
            import zhconv; return zhconv.convert(t,"zh-tw")
        except Exception: return t

def fmt_srt(s):
    ms=int(round(s%1*1000)); s=int(s)
    return f"{s//3600:02d}:{(s//60)%60:02d}:{s%60:02d},{ms:03d}"

def filter_segments(segs):
    kept,prev,run=[],None,0
    for s in segs:
        t=s.get("text","").strip()
        if not t or _HALL_RE.search(t): continue
        if t==prev:
            run+=1
            if run>=2: continue
        else:
            run,prev=0,t
            kept.append(s)
    _TRAIL={'完成','完','結束','结束','謝謝','谢谢','好','OK','ok','好的','完畢','end','END'}
    while kept and kept[-1].get("text","").strip() in _TRAIL and (kept[-1]['end']-kept[-1]['start'])<2.0:
        kept.pop()
    return kept

def build_srt(segs,to_tc=True,delay_ms=0):
    ss=[dict(s) for s in segs]
    if delay_ms:
        for s in ss: s["start"]=max(0,s["start"]+delay_ms/1000); s["end"]=max(s["start"]+.001,s["end"]+delay_ms/1000)
    ss.sort(key=lambda s:s["start"])
    bl,i=[],1
    for s in ss:
        t=(s.get("text")or"").strip()
        if not t: continue
        if to_tc: t=_to_tc(t)
        bl.append(f"{i}\n{fmt_srt(s['start'])} --> {fmt_srt(s['end'])}\n{t}"); i+=1
    return "\n\n".join(bl)+"\n\n" if bl else ""

def get_duration(path):
    try:
        r=subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
            "-of","default=noprint_wrappers=1:nokey=1",path],capture_output=True,text=True,timeout=30)
        return float(r.stdout.strip()) if r.stdout.strip() else 0.0
    except: return 0.0

# ── Whisper ───────────────────────────────────────────────────────────────
_cached_model=None; _cached_model_name=None
JOBS: dict={}

class _Cancel(BaseException): pass

class WhisperCapture:
    _RE=re.compile(r'\[(?:(\d+):)?(\d+):(\d+\.\d+)\s*-->\s*[^\]]+\]\s*(.*)')
    def __init__(self,dur,prog_cb,seg_cb,cancel_fn):
        self.dur,self.prog_cb,self.seg_cb,self.cancel_fn=max(dur,1),prog_cb,seg_cb,cancel_fn
        self._buf=""; self._ro,self._re=sys.__stdout__,sys.__stderr__
    def __enter__(self): self._ro,self._re=sys.stdout,sys.stderr; sys.stdout=sys.stderr=self; return self
    def __exit__(self,*a): sys.stdout,sys.stderr=self._ro,self._re; return False
    def flush(self): pass
    def write(self,text):
        if self.cancel_fn(): raise _Cancel()
        try: self._ro.write(text)
        except: pass
        self._buf+=text
        while "\n" in self._buf:
            line,self._buf=self._buf.split("\n",1)
            m=self._RE.match(line.strip())
            if m:
                cur=float(m.group(1)or 0)*3600+float(m.group(2))*60+float(m.group(3))
                self.prog_cb(min(95,int(cur/self.dur*100)))
                if m.group(4).strip(): self.seg_cb(line.strip(),m.group(4).strip())

def _transcribe_worker(jid,vpath,model,quality,prompt,to_tc,delay_ms):
    global _cached_model,_cached_model_name
    job=JOBS[jid]
    def log(m): job["log"].append(m)
    def prog(p): job["progress"]=p
    def seg(line,text): log(f"   {line[:60]}  →  {_to_tc(text) if to_tc else text}")
    try:
        import whisper
        if _cached_model_name!=model or _cached_model is None:
            log(f"📥 載入模型 {model}…"); prog(2)
            _cached_model=whisper.load_model(model); _cached_model_name=model
            log(f"✅ {model} 就緒")
        dur=get_duration(vpath); log(f"⏱ {dur:.1f}s"); prog(5)
        params=dict(QUALITY_PRESETS.get(quality,QUALITY_PRESETS["精準"]))
        if prompt: params["initial_prompt"]=prompt
        with WhisperCapture(dur,prog,seg,lambda:job.get("cancel",False)):
            result=_cached_model.transcribe(vpath,language="zh",task="transcribe",verbose=True,fp16=False,**params)
        segs=filter_segments(result.get("segments",[]))
        out=[]
        for s in segs:
            t=(s.get("text")or"").strip()
            if to_tc: t=_to_tc(t)
            ns={"start":s["start"],"end":s["end"],"text":t}
            if delay_ms: ns["start"]=max(0,ns["start"]+delay_ms/1000); ns["end"]=max(ns["start"]+.001,ns["end"]+delay_ms/1000)
            out.append(ns)
        log(f"✅ 完成 {len(out)} 段"); prog(100)
        job["segments"]=out; job["status"]="done"
    except _Cancel: log("⚠️ 已取消"); job["status"]="cancelled"
    except Exception as e: log(f"❌ {e}"); log(traceback.format_exc()); job["status"]="error"; job["error"]=str(e)

# ── Burn subtitles into video ─────────────────────────────────────────────
BURN_JOBS: dict={}

def _hex_to_ass(hex_color,alpha=1.0):
    h=hex_color.lstrip("#")
    r,g,b=int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
    a=int((1.0-alpha)*255)
    return f"&H{a:02X}{b:02X}{g:02X}{r:02X}"

def _sec_to_ass_ts(sec):
    h=int(sec//3600); m=int((sec%3600)//60); s=int(sec%60)
    cs=int(round((sec-int(sec))*100))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def _build_ass(segments,style):
    fname   = style.get("fontName","Arial")
    fsize   = int(style.get("fontSize",36))
    bold    = -1 if style.get("bold") else 0
    italic  = -1 if style.get("italic") else 0
    tcol    = _hex_to_ass(style.get("textColor","#ffffff"))
    bgst    = style.get("bgStyle","box")   # none/outline/box/solid
    bgcol   = style.get("bgColor","#000000")
    bgopa   = float(style.get("bgOpacity",0.75))
    xpct    = float(style.get("xPct",50))
    ypct    = float(style.get("yPct",90))
    # Use center-center anchor (\an5) + absolute \pos override for precise 2D placement
    px      = int(xpct/100*1920)
    py      = int(ypct/100*1080)
    pos_tag = f"{{\\an5\\pos({px},{py})}}"
    align   = 5   # center-center
    marginv = 0
    if bgst=="none":
        bs,outline,shadow,bcol=1,0,0,"&H00000000"
    elif bgst=="outline":
        bs,outline,shadow,bcol=1,2,1,_hex_to_ass(bgcol,bgopa)
    elif bgst=="solid":
        bs,outline,shadow,bcol=3,0,0,_hex_to_ass(bgcol,1.0)
    else:  # box
        bs,outline,shadow,bcol=3,0,0,_hex_to_ass(bgcol,bgopa)
    hdr=(
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        f"Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{fname},{fsize},{tcol},&H000000FF,&H00000000,{bcol},"
        f"{bold},{italic},0,0,100,100,0,0,{bs},{outline},{shadow},{align},10,10,{marginv},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events="\n".join(
        f"Dialogue: 0,{_sec_to_ass_ts(s['start'])},{_sec_to_ass_ts(s['end'])},Default,,0,0,0,,"
        f"{pos_tag}{s['text'].replace(chr(10),'\\N')}"
        for s in segments
    )
    return hdr+events+"\n"

def _burn_worker(bid,vpath,ass_path,out_path,crf):
    job=BURN_JOBS[bid]
    try:
        job["status"]="running"; job["progress"]=5
        dur=get_duration(vpath)
        cmd=["ffmpeg","-y","-i",vpath,"-vf",f"ass={ass_path}",
             "-c:v","libx264","-crf",str(crf),"-preset","fast","-c:a","copy",out_path]
        proc=subprocess.Popen(cmd,stderr=subprocess.PIPE,stdout=subprocess.DEVNULL,universal_newlines=True)
        tre=re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")
        for line in proc.stderr:
            if job.get("cancel"): proc.kill(); job["status"]="cancelled"; return
            m=tre.search(line)
            if m and dur>0:
                t=float(m.group(1))*3600+float(m.group(2))*60+float(m.group(3))
                job["progress"]=min(95,int(t/dur*100))
        proc.wait()
        if proc.returncode!=0: job["status"]="error"; job["error"]="ffmpeg 執行失敗（請確認已安裝 ffmpeg）"
        else: job["progress"]=100; job["status"]="done"; job["out"]=out_path
    except Exception as e: job["status"]="error"; job["error"]=str(e)

# ── Flask ─────────────────────────────────────────────────────────────────
app=Flask(__name__)

@app.route("/")
def index(): return HTML

@app.route("/api/models")
def api_models():
    cached=[n for n in MODELS if (WHISPER_CACHE/MODEL_FILES[n]).exists()]
    return jsonify({"models":MODELS,"cached":cached})

@app.route("/api/upload",methods=["POST"])
def api_upload():
    f=request.files.get("video")
    if not f: return jsonify({"error":"no file"}),400
    vid_id=str(uuid.uuid4())
    path=UPLOAD_DIR/f"{vid_id}{Path(f.filename).suffix.lower()}"
    f.save(path)
    return jsonify({"video_id":vid_id,"url":f"/uploads/{path.name}"})

@app.route("/uploads/<fn>")
def serve_video(fn): return send_from_directory(UPLOAD_DIR,fn)

@app.route("/api/transcribe",methods=["POST"])
def api_transcribe():
    d=request.json or {}
    matches=list(UPLOAD_DIR.glob(f"{d.get('video_id','')}.*"))
    if not matches: return jsonify({"error":"video not found"}),404
    jid=str(uuid.uuid4())
    JOBS[jid]={"status":"running","progress":0,"log":[],"segments":None,"error":None,"cancel":False}
    threading.Thread(target=_transcribe_worker,daemon=True,
        args=(jid,str(matches[0]),d.get("model","large-v2"),d.get("quality","精準"),
              d.get("prompt","繁體中文"),d.get("to_tc",True),d.get("delay_ms",0))).start()
    return jsonify({"job_id":jid})

@app.route("/api/job/<jid>")
def api_job(jid):
    j=JOBS.get(jid)
    if not j: return jsonify({"status":"not_found"}),404
    return jsonify({"status":j["status"],"progress":j["progress"],"log":j["log"][-30:],"segments":j["segments"],"error":j["error"]})

@app.route("/api/job/<jid>/cancel",methods=["POST"])
def api_cancel(jid):
    if jid in JOBS: JOBS[jid]["cancel"]=True
    return jsonify({"ok":True})

@app.route("/api/export_srt",methods=["POST"])
def api_export_srt():
    from io import BytesIO
    segs=(request.json or {}).get("segments",[])
    bio=BytesIO(build_srt(segs,to_tc=False).encode("utf-8")); bio.seek(0)
    return send_file(bio,as_attachment=True,download_name="subtitles.srt",mimetype="text/plain;charset=utf-8")

@app.route("/api/burn_start",methods=["POST"])
def api_burn_start():
    d=request.json or {}
    matches=list(UPLOAD_DIR.glob(f"{d.get('video_id','')}.*"))
    if not matches: return jsonify({"error":"video not found"}),404
    bid=str(uuid.uuid4())
    ass_path=str(UPLOAD_DIR/f"{bid}.ass")
    out_path=str(UPLOAD_DIR/f"{bid}_out.mp4")
    with open(ass_path,"w",encoding="utf-8") as f: f.write(_build_ass(d.get("segments",[]),d.get("style",{})))
    BURN_JOBS[bid]={"status":"pending","progress":0,"out":None,"error":None,"cancel":False}
    threading.Thread(target=_burn_worker,daemon=True,
        args=(bid,str(matches[0]),ass_path,out_path,int(d.get("crf",18)))).start()
    return jsonify({"burn_id":bid})

@app.route("/api/burn_job/<bid>")
def api_burn_job(bid):
    j=BURN_JOBS.get(bid)
    if not j: return jsonify({"status":"not_found"}),404
    return jsonify({"status":j["status"],"progress":j["progress"],"error":j["error"]})

@app.route("/api/burn_job/<bid>/cancel",methods=["POST"])
def api_burn_cancel(bid):
    if bid in BURN_JOBS: BURN_JOBS[bid]["cancel"]=True
    return jsonify({"ok":True})

@app.route("/api/burn_download/<bid>")
def api_burn_download(bid):
    j=BURN_JOBS.get(bid)
    if not j or j["status"]!="done" or not j.get("out"): return jsonify({"error":"not ready"}),404
    return send_file(j["out"],as_attachment=True,download_name="subtitled_video.mp4",mimetype="video/mp4")

# ── HTML ──────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>字幕編輯器</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0d1117;--p1:#161b22;--p2:#1c2333;--p3:#21262d;
  --bd:#30363d;--bd2:#21262d;
  --acc:#4493f8;--acc2:#79c0ff;--accH:#1f6feb;
  --txt:#e6edf3;--mut:#7d8590;--dim:#484f58;
  --grn:#3fb950;--yel:#d29922;--red:#f85149;--ora:#f0883e;
  --tl-sub:#0d9488;--tl-act:#f97316;
}
body{background:var(--bg);color:var(--txt);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden;font-size:13px}

/* ── Header ── */
header{display:flex;align-items:center;gap:6px;padding:6px 14px;background:var(--p1);border-bottom:1px solid var(--bd);flex-shrink:0;z-index:50}
.logo{font-size:15px;font-weight:800;color:var(--txt);letter-spacing:-.5px;margin-right:4px}
.logo b{color:var(--acc2)}
.sep{width:1px;height:18px;background:var(--bd);margin:0 3px;flex-shrink:0}
.spacer{flex:1}
.hbtn{padding:4px 11px;border-radius:6px;border:1px solid var(--bd);background:transparent;color:var(--mut);font-size:12px;font-weight:600;cursor:pointer;transition:all .14s;white-space:nowrap}
.hbtn:hover{border-color:var(--acc);color:var(--acc2);background:rgba(68,147,248,.08)}
.hbtn:disabled{opacity:.3;cursor:not-allowed;pointer-events:none}
.hbtn.pri{background:var(--acc);border-color:var(--acc);color:#fff}
.hbtn.pri:hover{background:var(--accH)}
.hbtn.active{border-color:var(--acc2);color:var(--acc2);background:rgba(68,147,248,.12)}
/* Export dropdown */
.exp-wrap{position:relative}
.exp-menu{position:absolute;top:calc(100% + 5px);right:0;background:var(--p1);border:1px solid var(--bd);border-radius:9px;overflow:hidden;z-index:200;min-width:135px;box-shadow:0 8px 24px rgba(0,0,0,.6);display:none}
.exp-menu.open{display:block}
.exp-menu button{display:block;width:100%;padding:9px 14px;border:none;background:transparent;color:var(--txt);font-size:12px;font-weight:600;cursor:pointer;text-align:left}
.exp-menu button:hover{background:var(--p3)}
.exp-menu .exp-sep{height:1px;background:var(--bd);margin:2px 0}

/* ── Content row ── */
.content-row{display:flex;flex:1;overflow:hidden;min-height:0}

/* ── Left panel ── */
.left-panel{flex:1;display:flex;flex-direction:column;background:#000;border-right:1px solid var(--bd);min-width:0}
.video-wrap{flex:1;position:relative;background:#000;display:flex;align-items:center;justify-content:center;overflow:hidden;min-height:0}
video{max-width:100%;max-height:100%;display:block}
.sub-overlay{position:absolute;text-align:center;pointer-events:none;max-width:90%;transition:left .07s,top .07s}
.upload-hint{display:flex;flex-direction:column;align-items:center;gap:14px;color:var(--dim)}
.upload-hint .ico{font-size:56px;opacity:.3}
.upload-hint p{font-size:13px;color:var(--mut)}
.ubtn{padding:10px 24px;background:var(--acc);border:none;border-radius:9px;color:#fff;font-size:14px;font-weight:700;cursor:pointer}
.ubtn:hover{background:var(--accH)}

/* Video controls */
.vc{background:var(--p1);border-top:1px solid var(--bd);padding:9px 12px;flex-shrink:0}
.seek-wrap{margin-bottom:8px;position:relative}
.seek{width:100%;height:4px;-webkit-appearance:none;appearance:none;background:var(--bd);border-radius:2px;cursor:pointer;outline:none}
.seek::-webkit-slider-thumb{-webkit-appearance:none;width:12px;height:12px;border-radius:50%;background:var(--acc2);cursor:pointer}
.vc-row{display:flex;align-items:center;gap:7px}
.vcb{background:transparent;border:1px solid var(--bd);border-radius:6px;color:var(--txt);width:32px;height:32px;display:flex;align-items:center;justify-content:center;cursor:pointer;font-size:13px;transition:all .13s;flex-shrink:0}
.vcb:hover{border-color:var(--acc);color:var(--acc2)}
.time-d{font-family:'SF Mono','Fira Code',monospace;font-size:11px;color:var(--mut);letter-spacing:.3px;flex-shrink:0}
.vol{display:flex;align-items:center;gap:5px;margin-left:auto;font-size:12px}
.vol input{width:60px;height:3px;cursor:pointer;accent-color:var(--acc2)}

/* Style strip */
.style-strip{background:var(--p2);border-top:1px solid var(--bd);padding:6px 12px;flex-shrink:0;display:none;gap:8px;align-items:center;flex-wrap:wrap}
.style-strip.show{display:flex}
.sg{display:flex;align-items:center;gap:5px}
.sg label{font-size:10px;font-weight:700;color:var(--dim);text-transform:uppercase;letter-spacing:.5px;white-space:nowrap}
.sg select,.sg input[type=number]{background:var(--p3);border:1px solid var(--bd);color:var(--txt);border-radius:5px;padding:3px 7px;font-size:12px;outline:none}
.sg select:focus,.sg input[type=number]:focus{border-color:var(--acc)}
.sg input[type=color]{width:26px;height:26px;border:1px solid var(--bd);border-radius:4px;cursor:pointer;padding:1px;background:var(--p3)}
.sg input[type=range]{width:70px;height:3px;cursor:pointer;accent-color:var(--acc2)}
.fmt-btn{width:26px;height:26px;border:1px solid var(--bd);border-radius:5px;background:var(--p3);color:var(--mut);font-size:12px;font-weight:700;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all .13s}
.fmt-btn:hover,.fmt-btn.on{border-color:var(--acc2);color:var(--acc2);background:rgba(68,147,248,.12)}
.bg-sel{display:flex;gap:3px}
.bg-opt{padding:3px 8px;border:1px solid var(--bd);border-radius:5px;background:var(--p3);color:var(--mut);font-size:11px;font-weight:600;cursor:pointer;transition:all .13s}
.bg-opt.on{border-color:var(--acc2);color:var(--acc2);background:rgba(68,147,248,.12)}
.pos-sel{display:flex;gap:3px}
.pos-btn{padding:3px 7px;border:1px solid var(--bd);border-radius:5px;background:var(--p3);color:var(--mut);font-size:11px;cursor:pointer;transition:all .13s}
.pos-btn.on{border-color:var(--acc2);color:var(--acc2);background:rgba(68,147,248,.12)}
.v-sep{width:1px;height:20px;background:var(--bd);flex-shrink:0}

/* ── Right panel ── */
.right-panel{width:380px;flex-shrink:0;display:flex;flex-direction:column;background:var(--p1);overflow:hidden}
.lhdr{display:flex;align-items:center;gap:6px;padding:8px 11px;border-bottom:1px solid var(--bd);flex-shrink:0}
.lhdr h3{font-size:13px;font-weight:700}
.lhdr .cnt{font-size:11px;color:var(--mut);padding:2px 7px;background:var(--p3);border-radius:99px}
.lhdr .spacer{flex:1}
.lhdr button{padding:3px 9px;border-radius:5px;border:1px solid var(--bd);background:var(--p3);color:var(--txt);font-size:12px;font-weight:600;cursor:pointer}
.lhdr button:hover{border-color:var(--acc);color:var(--acc2)}
.sub-list{flex:1;overflow-y:auto;padding:0}
.sub-list::-webkit-scrollbar{width:4px}
.sub-list::-webkit-scrollbar-thumb{background:var(--bd);border-radius:2px}

.srow{border-bottom:1px solid var(--bd2);padding:8px 10px;cursor:pointer;transition:background .1s;display:flex;flex-direction:column;gap:4px;border-left:3px solid transparent}
.srow:hover{background:rgba(68,147,248,.05)}
.srow.act{background:rgba(68,147,248,.1);border-left-color:var(--acc)}
.srow.shl{border-left-color:var(--yel);background:rgba(210,153,34,.06)}
.srow.scur{border-left-color:var(--yel);background:rgba(210,153,34,.14)!important}
.row-top{display:flex;align-items:center;gap:5px}
.sidx{font-size:11px;font-weight:700;color:var(--dim);width:22px;flex-shrink:0;text-align:right}
.stimes{display:flex;align-items:center;gap:4px;flex:1}
.ti{background:transparent;border:1px solid transparent;color:var(--acc2);font-family:'SF Mono','Fira Code',monospace;font-size:11px;font-weight:600;padding:2px 4px;border-radius:4px;width:100px;text-align:center;cursor:pointer;outline:none;transition:all .13s}
.ti:hover{border-color:var(--bd);background:var(--p3)}
.ti:focus{border-color:var(--acc);background:var(--p3);color:#fff}
.tarr{color:var(--dim);font-size:11px;flex-shrink:0}
.sact{display:flex;gap:2px;margin-left:auto}
.sact button{background:transparent;border:1px solid transparent;border-radius:4px;color:var(--dim);font-size:12px;width:22px;height:22px;display:flex;align-items:center;justify-content:center;cursor:pointer;transition:all .1s}
.sact button:hover{background:var(--p3);border-color:var(--bd);color:var(--txt)}
.sact button.dl:hover{color:var(--red);border-color:var(--red)}
.stxt{width:100%;background:transparent;border:1px solid transparent;color:var(--txt);font-size:13px;font-weight:400;padding:4px 7px;border-radius:5px;resize:none;outline:none;line-height:1.55;transition:all .13s;font-family:inherit;min-height:32px}
.stxt:hover{border-color:var(--bd);background:var(--p3)}
.stxt:focus{border-color:var(--acc);background:var(--p3)}

/* Transcribe section inside right panel */
.tbar{background:var(--p2);border-top:1px solid var(--bd);padding:9px 12px;flex-shrink:0}
.tbar-toggle{display:flex;align-items:center;gap:6px;cursor:pointer;margin-bottom:0;user-select:none}
.tbar-toggle span{font-size:11px;font-weight:700;color:var(--mut);text-transform:uppercase;letter-spacing:.5px;flex:1}
.tbar-toggle .arr{color:var(--dim);font-size:11px;transition:transform .2s}
.tbar-toggle .arr.open{transform:rotate(180deg)}
.tbar-body{margin-top:9px;display:none}
.tbar-body.open{display:block}
.tbar-row1{display:flex;align-items:center;gap:7px;margin-bottom:7px;flex-wrap:wrap}
.tbar-row1 label{font-size:11px;font-weight:700;color:var(--dim);text-transform:uppercase;letter-spacing:.4px}
.tbar-row1 select,.tbar-row1 input[type=text]{background:var(--p3);border:1px solid var(--bd);color:var(--txt);border-radius:5px;padding:4px 8px;font-size:12px;outline:none}
.tbar-row1 select:focus,.tbar-row1 input[type=text]:focus{border-color:var(--acc)}
.pinp{flex:1;min-width:100px}
.tbar-row2{display:flex;align-items:center;gap:7px}
.pw{flex:1}
.pbg{height:4px;background:var(--p3);border-radius:2px;overflow:hidden}
.pfill{height:100%;background:var(--acc);border-radius:2px;transition:width .3s;width:0%}
.plbl{font-size:11px;color:var(--mut);margin-top:3px}
.tbtns{display:flex;gap:6px}
.tbtns button{padding:5px 13px;border-radius:7px;border:none;font-size:12px;font-weight:700;cursor:pointer}
.btn-go{background:var(--acc);color:#fff}
.btn-go:hover{background:var(--accH)}
.btn-go:disabled{background:var(--dim);cursor:not-allowed}
.btn-stop{background:rgba(248,81,73,.12);color:var(--red);border:1px solid rgba(248,81,73,.3)!important}
.btn-stop:hover{background:rgba(248,81,73,.22)}
.sdot{width:7px;height:7px;border-radius:50%;background:var(--dim);flex-shrink:0}
.sdot.run{background:var(--acc);animation:pulse 1s infinite}
.sdot.done{background:var(--grn)}
.sdot.error{background:var(--red)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.logbox{max-height:65px;overflow-y:auto;background:var(--bg);border-radius:5px;padding:5px 8px;font-family:'SF Mono','Fira Code',monospace;font-size:10px;color:var(--mut);line-height:1.6;margin-top:6px;display:none}
.logbox.show{display:block}

/* ── Full-width Timeline ── */
.tl-section{flex-shrink:0;height:90px;background:#0a0f1a;border-top:1px solid var(--bd);position:relative;overflow:hidden}
.tl-toolbar{position:absolute;top:3px;right:5px;z-index:3;display:flex;gap:3px}
.tlb{padding:2px 7px;border-radius:4px;border:1px solid var(--bd);background:rgba(13,17,23,.88);color:var(--dim);font-size:10px;font-weight:600;cursor:pointer;transition:all .12s}
.tlb:hover{border-color:var(--acc2);color:var(--acc2)}
#tlCv{display:block;width:100%;height:100%;cursor:crosshair}

/* ── Modals ── */
.mover{position:fixed;inset:0;background:rgba(0,0,0,.65);backdrop-filter:blur(4px);z-index:300;display:none;align-items:center;justify-content:center}
.mover.open{display:flex}
.modal{background:var(--p1);border:1px solid var(--bd);border-radius:12px;padding:20px;width:460px;max-width:92vw;max-height:88vh;overflow-y:auto}
.modal h3{font-size:14px;font-weight:800;margin-bottom:13px}
.modal textarea{width:100%;height:170px;background:var(--bg);border:1px solid var(--bd);border-radius:7px;color:var(--txt);font-family:'SF Mono','Fira Code',monospace;font-size:12px;padding:8px;resize:vertical;outline:none}
.modal textarea:focus{border-color:var(--acc)}
.mbtns{display:flex;gap:7px;margin-top:12px;justify-content:flex-end;flex-wrap:wrap}
.mbtns button{padding:6px 15px;border-radius:7px;border:1px solid var(--bd);background:var(--p3);color:var(--txt);font-size:12px;font-weight:600;cursor:pointer}
.mbtns button:hover{border-color:var(--acc)}
.mbtns button.pri{background:var(--acc);border-color:var(--acc);color:#fff}
.mbtns button.pri:hover{background:var(--accH)}
/* Search */
.sf{display:flex;align-items:center;gap:7px;margin-bottom:9px}
.sf label{font-size:12px;color:var(--mut);width:32px;text-align:right;flex-shrink:0}
.sf input{flex:1;background:var(--bg);border:1px solid var(--bd);color:var(--txt);border-radius:6px;padding:5px 9px;font-size:12px;outline:none}
.sf input:focus{border-color:var(--acc)}
.snav{display:flex;align-items:center;gap:5px;margin-top:4px}
.snav button{padding:4px 9px;border-radius:5px;border:1px solid var(--bd);background:var(--p3);color:var(--txt);font-size:12px;cursor:pointer}
.snav button:hover{border-color:var(--acc)}
.scnt{font-size:12px;color:var(--mut);min-width:65px;text-align:center}
/* Burn modal */
.burn-prog{height:5px;background:var(--p3);border-radius:3px;overflow:hidden;margin-top:10px;display:none}
.burn-fill{height:100%;background:var(--grn);border-radius:3px;transition:width .3s;width:0%}
.burn-status{font-size:12px;color:var(--mut);margin-top:6px}
/* Shift */
.shift-inp{width:100%;margin-top:8px;font-size:15px;text-align:center;padding:8px;border-radius:7px;background:var(--p3);border:1px solid var(--bd);color:var(--txt);outline:none}
.shift-inp:focus{border-color:var(--acc)}
/* Loading overlay */
#loadOv{position:fixed;inset:0;background:rgba(0,0,0,.85);backdrop-filter:blur(8px);z-index:1000;display:none;flex-direction:column;align-items:center;justify-content:center;gap:16px}
#loadOv.on{display:flex}
.ld-spin{width:54px;height:54px;border:5px solid var(--bd);border-top-color:var(--acc);border-radius:50%;animation:spin .75s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.ld-msg{font-size:17px;font-weight:700;color:var(--txt);letter-spacing:-.3px}
.ld-sub{font-size:13px;color:var(--mut);min-height:18px}
.ld-bar{width:300px;height:7px;background:var(--p3);border-radius:4px;overflow:hidden}
.ld-fill{height:100%;background:linear-gradient(90deg,var(--acc),var(--acc2));border-radius:4px;transition:width .35s;width:0%}
.ld-cancel{padding:7px 22px;border-radius:8px;border:1px solid var(--bd);background:transparent;color:var(--mut);font-size:13px;font-weight:600;cursor:pointer;margin-top:6px;transition:all .15s}
.ld-cancel:hover{border-color:var(--red);color:var(--red)}
/* Timeline zoom slider */
.tl-zoom-row{display:flex;align-items:center;gap:5px}
.tl-zoom-lbl{font-size:10px;color:var(--dim)}
#tlZoomBar{width:80px;height:3px;cursor:pointer;accent-color:var(--acc2);vertical-align:middle}
#tlZoomPct{font-size:10px;color:var(--mut);min-width:36px;display:inline-block;text-align:right}
/* Drag guide lines */
.gl-h{position:absolute;left:0;right:0;height:0;border-top:1px dashed rgba(34,211,238,0);pointer-events:none;transition:border-color .1s}
.gl-v{position:absolute;top:0;bottom:0;width:0;border-left:1px dashed rgba(34,211,238,0);pointer-events:none;transition:border-color .1s}
.gl-h.snap{border-color:rgba(34,211,238,.85)}
.gl-v.snap{border-color:rgba(34,211,238,.85)}
/* Sub span drag cursor */
#subOv span{cursor:move}
</style>
</head>
<body>

<!-- Header -->
<header>
  <div class="logo">字幕<b>編輯器</b></div>
  <button class="hbtn" onclick="document.getElementById('srtIn').click()">📂 匯入</button>
  <button class="hbtn" onclick="openPaste()">📋 貼上</button>
  <div class="sep"></div>
  <button class="hbtn" onclick="openSearch()" title="Ctrl+F">🔍 搜尋</button>
  <button class="hbtn" onclick="openShift()">⏱ 偏移</button>
  <button class="hbtn" id="btnStyleToggle" onclick="toggleStyle()">✏️ 字幕樣式</button>
  <div class="sep"></div>
  <button class="hbtn" id="btnU" onclick="undo()" disabled>⎌ 復原</button>
  <button class="hbtn" id="btnR" onclick="redo()" disabled>⇧⎌ 重做</button>
  <div class="spacer"></div>
  <button class="hbtn" id="btnBurnH" onclick="openBurn()" style="background:rgba(194,65,12,.18);border-color:#f97316;color:#fb923c;font-weight:700">🎬 燒錄匯出</button>
  <div class="exp-wrap">
    <button class="hbtn pri" onclick="toggleExp()">⬇ 匯出 ▾</button>
    <div class="exp-menu" id="expMenu">
      <button onclick="exportSRT();closeExp()">SRT 格式</button>
      <button onclick="exportVTT();closeExp()">VTT 格式</button>
      <button onclick="exportASS();closeExp()">ASS 格式</button>
    </div>
  </div>
</header>

<!-- Content row -->
<div class="content-row">

  <!-- Left: video -->
  <div class="left-panel">
    <div class="video-wrap" id="vWrap">
      <div class="upload-hint" id="uHint">
        <div class="ico">🎬</div>
        <p>拖曳影片到這裡，或</p>
        <button class="ubtn" onclick="document.getElementById('vidIn').click()">選擇影片檔案</button>
        <p style="font-size:11px">MP4、MKV、MOV、AVI…</p>
      </div>
      <video id="vid" style="display:none"></video>
      <!-- Snap guide lines (shown during drag) -->
      <div id="guideOv" style="position:absolute;inset:0;pointer-events:none;z-index:9;display:none">
        <div class="gl-h" id="glTH" style="top:10%"></div>
        <div class="gl-h" id="glCH" style="top:50%"></div>
        <div class="gl-h" id="glBH" style="top:90%"></div>
        <div class="gl-v" id="glLV" style="left:10%"></div>
        <div class="gl-v" id="glCV" style="left:50%"></div>
        <div class="gl-v" id="glRV" style="left:90%"></div>
      </div>
      <div class="sub-overlay" id="subOv"></div>
    </div>

    <div class="vc" id="vcArea" style="display:none">
      <div class="seek-wrap">
        <input type="range" class="seek" id="seekBar" min="0" max="100" step="0.01" value="0">
      </div>
      <div class="vc-row">
        <button class="vcb" id="btnPlay" onclick="togglePlay()">▶</button>
        <button class="vcb" onclick="skip(-5)">-5s</button>
        <button class="vcb" onclick="skip(5)">+5s</button>
        <span class="time-d" id="timeD">00:00:00 / 00:00:00</span>
        <div class="vol">🔊<input type="range" min="0" max="1" step="0.05" value="1" oninput="vid.volume=this.value"></div>
      </div>
    </div>

    <!-- Style strip -->
    <div class="style-strip" id="styleStrip">
      <div class="sg"><label>字體</label>
        <select id="sFnt" onchange="updStyle()" style="max-width:180px">
          <optgroup label="── macOS 繁中 ──">
            <option value="PingFang TC">蘋方繁體 PingFang TC</option>
            <option value="Heiti TC">黑體繁體 Heiti TC</option>
            <option value="Songti TC">宋體繁體 Songti TC</option>
            <option value="Kaiti TC">楷體繁體 Kaiti TC</option>
            <option value="LiHei Pro">儷黑 Pro</option>
            <option value="LiSong Pro">儷宋 Pro</option>
            <option value="BiauKai">標楷體 BiauKai</option>
            <option value="WeiBei TC">娃娃體 WeiBei TC</option>
          </optgroup>
          <optgroup label="── Google／Adobe ──">
            <option value="Noto Sans TC">Noto Sans TC</option>
            <option value="Noto Serif TC">Noto Serif TC</option>
            <option value="Source Han Sans TW">思源黑體 TW</option>
          </optgroup>
          <optgroup label="── Windows 繁中 ──">
            <option value="Microsoft JhengHei">微軟正黑體</option>
            <option value="PMingLiU">新細明體 PMingLiU</option>
          </optgroup>
          <optgroup label="── Western ──">
            <option value="Arial" selected>Arial</option>
            <option value="Helvetica Neue">Helvetica Neue</option>
            <option value="Impact">Impact</option>
            <option value="Georgia">Georgia</option>
          </optgroup>
        </select>
      </div>
      <div class="sg"><label>大小</label>
        <input type="number" id="sFsz" value="36" min="12" max="120" style="width:55px" onchange="updStyle()">
      </div>
      <div class="sg"><label>樣式</label>
        <button class="fmt-btn" id="fmtB" onclick="toggleFmt('bold')" title="粗體">B</button>
        <button class="fmt-btn" id="fmtI" onclick="toggleFmt('italic')" title="斜體"><i>I</i></button>
      </div>
      <div class="sg"><label>文字色</label>
        <input type="color" id="sTxt" value="#ffffff" onchange="updStyle()">
      </div>
      <div class="v-sep"></div>
      <div class="sg"><label>背景</label>
        <div class="bg-sel">
          <button class="bg-opt on" data-v="box"     onclick="setBg(this,'box')">半透明框</button>
          <button class="bg-opt"    data-v="outline"  onclick="setBg(this,'outline')">描邊</button>
          <button class="bg-opt"    data-v="solid"    onclick="setBg(this,'solid')">實心框</button>
          <button class="bg-opt"    data-v="none"     onclick="setBg(this,'none')">無</button>
        </div>
      </div>
      <div class="sg" id="bgColG"><label>背景色</label>
        <input type="color" id="sBg" value="#000000" onchange="updStyle()">
      </div>
      <div class="sg" id="bgOpaG"><label>透明度</label>
        <input type="range" id="sOpa" min="0" max="1" step="0.05" value="0.75" oninput="updStyle()">
      </div>
      <div class="v-sep"></div>
      <div class="sg"><label>快速位置</label>
        <div class="pos-sel">
          <button class="pos-btn" data-v="top"    onclick="setPos(this,'top')">▲ 頂</button>
          <button class="pos-btn" data-v="middle" onclick="setPos(this,'middle')">● 中</button>
          <button class="pos-btn on" data-v="bottom" onclick="setPos(this,'bottom')">▼ 底</button>
        </div>
      </div>
      <div class="sg"><label>座標</label>
        <span id="posReadout" style="font-size:11px;color:var(--acc2);font-family:'SF Mono','Fira Code',monospace;letter-spacing:.3px">X:50% Y:90%</span>
      </div>
    </div>
  </div>

  <!-- Right: subtitle list + transcribe -->
  <div class="right-panel">
    <div class="lhdr">
      <h3>字幕列表</h3>
      <span class="cnt" id="subCnt">0 段</span>
      <div class="spacer"></div>
      <button onclick="addSub()">＋ 新增</button>
    </div>
    <div class="sub-list" id="subList">
      <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:180px;gap:8px;color:var(--dim)">
        <div style="font-size:34px;opacity:.25">📝</div>
        <p>匯入 SRT 或先轉錄影片</p>
      </div>
    </div>

    <!-- Transcribe -->
    <div class="tbar">
      <div class="tbar-toggle" onclick="toggleTbar()">
        <span>🎙 轉錄設定</span>
        <span class="arr open" id="tbarArr">▼</span>
      </div>
      <div class="tbar-body open" id="tbarBody">
        <div class="tbar-row1">
          <label>模型</label>
          <select id="mdlSel">
            <option value="tiny">tiny 極快</option>
            <option value="base">base 快速</option>
            <option value="small">small</option>
            <option value="medium">medium</option>
            <option value="large-v2" selected>large-v2 ⭐</option>
            <option value="large-v3">large-v3 最新</option>
          </select>
          <label>品質</label>
          <select id="qSel"><option value="精準" selected>精準</option><option value="快速">快速</option></select>
        </div>
        <div class="tbar-row1">
          <label>提示</label>
          <input type="text" id="prompt" class="pinp" value="繁體中文">
          <label><input type="checkbox" id="toTC" checked> 繁</label>
          <label>偏移ms<input type="number" id="dms" value="0" style="width:55px;margin-left:4px"></label>
        </div>
        <div class="tbar-row2">
          <div class="sdot" id="sDot"></div>
          <div class="pw">
            <div class="pbg"><div class="pfill" id="pFill"></div></div>
            <div class="plbl" id="pLbl">等待轉錄…</div>
          </div>
          <div class="tbtns">
            <button class="btn-go" id="btnGo" onclick="startTx()">▶ 開始轉錄</button>
            <button class="btn-stop" id="btnStop" onclick="cancelTx()" style="display:none">⏹ 取消</button>
          </div>
        </div>
        <div class="logbox" id="logBox"></div>
      </div>
    </div>
  </div>
</div>

<!-- Timeline (full width) -->
<div class="tl-section" id="tlSec" style="display:none">
  <div class="tl-toolbar" style="display:flex;align-items:center;gap:5px">
    <span class="tl-zoom-lbl">縮放</span>
    <input type="range" id="tlZoomBar" min="0" max="100" value="0" oninput="tlZoomSlide(this.value)">
    <span id="tlZoomPct">100%</span>
    <button class="tlb" onclick="tlFit()">全覽</button>
    <button class="tlb" id="btnWf" onclick="loadWf()">📊 波形</button>
  </div>
  <canvas id="tlCv"></canvas>
</div>

<!-- Loading overlay -->
<div id="loadOv">
  <div class="ld-spin"></div>
  <div class="ld-msg" id="ldMsg">載入中…</div>
  <div class="ld-sub" id="ldSub"></div>
  <div class="ld-bar"><div class="ld-fill" id="ldFill"></div></div>
  <button class="ld-cancel" id="ldCancel" style="display:none" onclick="ldCancelFn&&ldCancelFn()">取消</button>
</div>

<!-- Hidden inputs -->
<input type="file" id="vidIn" accept="video/*" style="display:none" onchange="onVidPick(this)">
<input type="file" id="srtIn" accept=".srt,.txt" style="display:none" onchange="onSrtPick(this)">

<!-- Paste modal -->
<div class="mover" id="pasteM">
  <div class="modal">
    <h3>📋 貼上 SRT 內容</h3>
    <textarea id="pasteA" placeholder="把 SRT 字幕貼在這裡…"></textarea>
    <div class="mbtns"><button onclick="closePaste()">取消</button><button class="pri" onclick="doPaste()">匯入</button></div>
  </div>
</div>

<!-- Shift modal -->
<div class="mover" id="shiftM">
  <div class="modal" style="width:320px">
    <h3>⏱ 整體時間偏移</h3>
    <p style="font-size:12px;color:var(--mut);margin-bottom:8px">毫秒。正數＝延後，負數＝提前</p>
    <input type="number" class="shift-inp" id="shiftV" value="0">
    <div class="mbtns"><button onclick="closeShift()">取消</button><button class="pri" onclick="doShift()">套用</button></div>
  </div>
</div>

<!-- Search modal -->
<div class="mover" id="searchM">
  <div class="modal">
    <h3>🔍 搜尋與取代</h3>
    <div class="sf"><label>搜尋</label><input type="text" id="sQ" placeholder="搜尋文字…" oninput="doSearch()"></div>
    <div class="sf"><label>取代</label><input type="text" id="sR" placeholder="取代成…"></div>
    <div class="snav">
      <label style="font-size:12px;color:var(--mut)"><input type="checkbox" id="sCS"> 大小寫</label>
      <div style="flex:1"></div>
      <button onclick="srPrev()">↑ 上一個</button>
      <span class="scnt" id="sCnt">–</span>
      <button onclick="srNext()">↓ 下一個</button>
    </div>
    <div class="mbtns">
      <button onclick="closeSearch()">關閉</button>
      <button onclick="replOne()">取代此筆</button>
      <button class="pri" onclick="replAll()">全部取代</button>
    </div>
  </div>
</div>

<!-- Burn modal -->
<div class="mover" id="burnM">
  <div class="modal">
    <h3>🎬 燒錄字幕到影片</h3>
    <p style="font-size:12px;color:var(--mut);margin-bottom:12px">將目前字幕與樣式設定嵌入影片。使用 FFmpeg 重新編碼，視覺品質接近無損。</p>
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
      <label style="font-size:12px;color:var(--mut)">品質</label>
      <select id="burnCrf" style="background:var(--p3);border:1px solid var(--bd);color:var(--txt);border-radius:5px;padding:4px 8px;font-size:12px;outline:none">
        <option value="15">極高品質（CRF 15，檔案較大）</option>
        <option value="18" selected>高品質（CRF 18，推薦）</option>
        <option value="23">一般品質（CRF 23，較小）</option>
      </select>
    </div>
    <div class="mbtns">
      <button onclick="closeBurn()">取消</button>
      <button class="pri" id="btnBurn" onclick="startBurn()">🎬 開始燒錄</button>
    </div>
  </div>
</div>

<script>
// ── State ────────────────────────────────────────────────────────────────
const vid=document.getElementById('vid');
let subs=[], activeIdx=-1, videoId=null, jobId=null, pollTmr=null;

// ── Subtitle style state ──────────────────────────────────────────────────
let st={fontName:'Arial',fontSize:36,bold:false,italic:false,
        textColor:'#ffffff',bgStyle:'box',bgColor:'#000000',bgOpacity:.75,
        xPct:50,yPct:90};

// ── Time helpers ──────────────────────────────────────────────────────────
function s2srt(s){
  s=Math.max(0,s);
  const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),sec=Math.floor(s%60),ms=Math.round((s-Math.floor(s))*1000);
  return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')},${String(ms).padStart(3,'0')}`;
}
function srt2s(t){const[h,m,r]=t.split(':');const[sec,ms]=r.replace(',','.').split('.');return parseInt(h)*3600+parseInt(m)*60+parseInt(sec)+(parseInt(ms)||0)/1000;}
function s2hms(s){const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),sec=Math.floor(s%60);return`${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')}`;}
function s2ass(s){const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),sec=Math.floor(s%60),cs=Math.round((s-Math.floor(s))*100);return`${h}:${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')}.${String(cs).padStart(2,'0')}`;}
function hex2rgba(h,a){const r=parseInt(h.slice(1,3),16),g=parseInt(h.slice(3,5),16),b=parseInt(h.slice(5,7),16);return`rgba(${r},${g},${b},${a})`;}

// ── SRT parser ────────────────────────────────────────────────────────────
function parseSRT(txt){
  const res=[];
  for(const blk of txt.trim().split(/\n\n+/)){
    const ls=blk.trim().split('\n');
    const tl=ls.find(l=>/\d{2}:\d{2}:\d{2}[,\.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,\.]\d{3}/.test(l));
    if(!tl)continue;
    const m=tl.match(/(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})/);
    if(!m)continue;
    const ti=ls.indexOf(tl)+1;
    res.push({start:srt2s(m[1].replace('.',',')),end:srt2s(m[2].replace('.',',')),text:ls.slice(ti).join('\n').trim()});
  }
  return res.sort((a,b)=>a.start-b.start);
}

// ── Undo/Redo ────────────────────────────────────────────────────────────
let uStack=[],rStack=[];
function push(){uStack.push(JSON.stringify(subs));if(uStack.length>80)uStack.shift();rStack=[];updUR();}
function undo(){if(!uStack.length)return;rStack.push(JSON.stringify(subs));subs=JSON.parse(uStack.pop());activeIdx=-1;renderList();renderTL();updUR();}
function redo(){if(!rStack.length)return;uStack.push(JSON.stringify(subs));subs=JSON.parse(rStack.pop());activeIdx=-1;renderList();renderTL();updUR();}
function updUR(){document.getElementById('btnU').disabled=!uStack.length;document.getElementById('btnR').disabled=!rStack.length;}

// ── Style controls ────────────────────────────────────────────────────────
function toggleStyle(){
  const s=document.getElementById('styleStrip');
  const b=document.getElementById('btnStyleToggle');
  s.classList.toggle('show');
  b.classList.toggle('active',s.classList.contains('show'));
}
function updStyle(){
  st.fontName =document.getElementById('sFnt').value;
  st.fontSize =parseInt(document.getElementById('sFsz').value)||36;
  st.textColor=document.getElementById('sTxt').value;
  st.bgColor  =document.getElementById('sBg').value;
  st.bgOpacity=parseFloat(document.getElementById('sOpa').value);
  applyOvStyle();
}
function toggleFmt(k){
  st[k]=!st[k];
  document.getElementById(k==='bold'?'fmtB':'fmtI').classList.toggle('on',st[k]);
  applyOvStyle();
}
function setBg(el,v){
  document.querySelectorAll('.bg-opt').forEach(b=>b.classList.remove('on'));
  el.classList.add('on'); st.bgStyle=v;
  const hasColor=v!=='none';
  document.getElementById('bgColG').style.display=hasColor?'':'none';
  document.getElementById('bgOpaG').style.display=(v==='box'||v==='outline')?'':'none';
  applyOvStyle();
}
// Position shortcuts: snap to standard positions
const _POS_MAP={top:{xPct:50,yPct:10},middle:{xPct:50,yPct:50},bottom:{xPct:50,yPct:90}};
function setPos(el,v){
  document.querySelectorAll('.pos-btn').forEach(b=>b.classList.remove('on'));
  el.classList.add('on');
  const p=_POS_MAP[v]||_POS_MAP.bottom;
  st.xPct=p.xPct; st.yPct=p.yPct;
  updatePosReadout(); applyOvStyle();
}
function updatePosReadout(){
  const el=document.getElementById('posReadout');
  if(el) el.textContent=`X:${Math.round(st.xPct)}% Y:${Math.round(st.yPct)}%`;
}
function applyOvStyle(){
  const ov=document.getElementById('subOv');
  ov.style.left=st.xPct+'%';
  ov.style.top=st.yPct+'%';
  ov.style.transform='translate(-50%,-50%)';
  ov.style.bottom='auto';
  window._spanSt=buildSpanSt();
  const t=activeIdx>=0&&subs[activeIdx]?subs[activeIdx].text:'';
  renderOv(t);
}
function buildSpanSt(){
  let s=`font-family:${st.fontName};font-size:${st.fontSize}px;color:${st.textColor};`;
  s+=`font-weight:${st.bold?700:400};font-style:${st.italic?'italic':'normal'};`;
  if(st.bgStyle==='none'){s+='background:transparent;padding:2px 8px;text-shadow:1px 1px 3px #000,-1px -1px 3px #000;';}
  else if(st.bgStyle==='outline'){s+=`background:transparent;padding:2px 8px;text-shadow:1px 1px 3px #000,-1px -1px 3px #000,1px -1px 3px #000,-1px 1px 3px #000,0 0 8px #000;`;}
  else if(st.bgStyle==='solid'){s+=`background:${st.bgColor};padding:4px 14px;border-radius:4px;`;}
  else{s+=`background:${hex2rgba(st.bgColor,st.bgOpacity)};padding:4px 14px;border-radius:6px;`;}
  return s;
}
function renderOv(text){
  const ov=document.getElementById('subOv');
  if(!text){ov.innerHTML='';return;}
  const spanSt=window._spanSt||'background:rgba(0,0,0,.78);color:#fff;font-size:17px;font-weight:700;padding:4px 14px;border-radius:6px;';
  ov.innerHTML=`<span draggable="false" style="${spanSt};display:inline-block;line-height:1.5;white-space:pre-wrap;text-align:center;pointer-events:auto;user-select:none">${esc(text)}</span>`;
  // Re-attach drag listener every render (innerHTML replaces DOM)
  ov.querySelector('span').addEventListener('mousedown',_subSpanDown);
}
window._spanSt=buildSpanSt();
applyOvStyle();  // set initial position (xPct:50 yPct:90)

// ── Subtitle overlay 2D drag + guide snapping ────────────────────────────
let ovDragSX=0,ovDragSY=0,ovDragX0=50,ovDragY0=90;
const GUIDES_X=[10,50,90], GUIDES_Y=[10,50,90], SNAP_GT=5;

function _snapG(val,guides,thresh){
  let best=val,bestD=thresh,snapped=null;
  guides.forEach(g=>{const d=Math.abs(val-g);if(d<bestD){bestD=d;best=g;snapped=g;}});
  return{val:best,snapped};
}
function _showGuides(sx,sy){
  document.getElementById('guideOv').style.display='block';
  document.getElementById('glTH').classList.toggle('snap',sy===10);
  document.getElementById('glCH').classList.toggle('snap',sy===50);
  document.getElementById('glBH').classList.toggle('snap',sy===90);
  document.getElementById('glLV').classList.toggle('snap',sx===10);
  document.getElementById('glCV').classList.toggle('snap',sx===50);
  document.getElementById('glRV').classList.toggle('snap',sx===90);
}
function _hideGuides(){
  document.getElementById('guideOv').style.display='none';
  document.querySelectorAll('.gl-h,.gl-v').forEach(el=>el.classList.remove('snap'));
}
function _subSpanDown(e){
  if(!vid.src)return;
  ovDrag=true; ovDragSX=e.clientX; ovDragSY=e.clientY;
  ovDragX0=st.xPct; ovDragY0=st.yPct;
  e.preventDefault(); e.stopPropagation();
}

// ── Loading overlay ────────────────────────────────────────────────────────
let ldCancelFn=null;
function showLoading(msg,sub,showCancel,cancelFn){
  document.getElementById('ldMsg').textContent=msg||'載入中…';
  document.getElementById('ldSub').textContent=sub||'';
  document.getElementById('ldFill').style.width='0%';
  document.getElementById('ldCancel').style.display=showCancel?'':'none';
  ldCancelFn=cancelFn||null;
  document.getElementById('loadOv').classList.add('on');
}
function updateLoading(pct,sub){
  document.getElementById('ldFill').style.width=(pct||0)+'%';
  if(sub!=null)document.getElementById('ldSub').textContent=sub;
}
function hideLoading(){
  document.getElementById('loadOv').classList.remove('on');
  ldCancelFn=null;
}

// ── Subtitle list ─────────────────────────────────────────────────────────
function esc(t){return String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function ar(el){el.style.height='auto';el.style.height=el.scrollHeight+'px';}

function renderList(){
  const L=document.getElementById('subList');
  document.getElementById('subCnt').textContent=subs.length+' 段';
  if(!subs.length){
    L.innerHTML=`<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:180px;gap:8px;color:var(--dim)"><div style="font-size:34px;opacity:.25">📝</div><p>尚無字幕</p></div>`;
    return;
  }
  L.innerHTML=subs.map((s,i)=>`
    <div class="srow ${i===activeIdx?'act':''}" id="row-${i}" onclick="selSub(${i})">
      <div class="row-top">
        <span class="sidx">${i+1}</span>
        <div class="stimes">
          <input class="ti" id="ts-${i}" value="${s2srt(s.start)}" onclick="event.stopPropagation()" onfocus="this.select()" onchange="onTC(${i},'start',this.value)">
          <span class="tarr">→</span>
          <input class="ti" id="te-${i}" value="${s2srt(s.end)}" onclick="event.stopPropagation()" onfocus="this.select()" onchange="onTC(${i},'end',this.value)">
        </div>
        <div class="sact">
          <button onclick="event.stopPropagation();splitSub(${i})" title="分割">✂</button>
          <button onclick="event.stopPropagation();mergeUp(${i})" title="與上段合併" ${i===0?'disabled':''}>⬆</button>
          <button onclick="event.stopPropagation();addAfter(${i})" title="下方新增">＋</button>
          <button class="dl" onclick="event.stopPropagation();delSub(${i})" title="刪除">✕</button>
        </div>
      </div>
      <textarea class="stxt" id="tx-${i}" rows="2"
        onclick="event.stopPropagation()"
        onchange="onTxt(${i},this.value)"
        oninput="ar(this)"
      >${esc(s.text)}</textarea>
    </div>`).join('');
  L.querySelectorAll('.stxt').forEach(ar);
}

function selSub(i){
  activeIdx=i;
  if(subs[i]&&!isNaN(subs[i].start)) vid.currentTime=subs[i].start;
  document.querySelectorAll('.srow').forEach((r,j)=>r.classList.toggle('act',j===i));
  renderTL();
}
function onTC(i,f,v){push();try{const s=srt2s(v);if(!isNaN(s)){subs[i][f]=s;renderTL();}}catch(e){}}
function onTxt(i,v){push();subs[i].text=v;}

function addSub(){
  push();
  const last=subs.length?subs[subs.length-1].end:vid.currentTime||0;
  subs.push({start:last+.2,end:last+2.2,text:''});
  renderList();renderTL();
  setTimeout(()=>{const i=subs.length-1;document.getElementById(`row-${i}`)?.scrollIntoView({block:'nearest'});document.getElementById(`tx-${i}`)?.focus();},50);
}
function addAfter(i){
  push();
  const c=subs[i],n=subs[i+1],g=n?n.start:c.end+3,s=c.end+.2,e=Math.min(s+2,g-.1);
  subs.splice(i+1,0,{start:s,end:e,text:''});
  renderList();renderTL();
  setTimeout(()=>{document.getElementById(`tx-${i+1}`)?.focus();document.getElementById(`row-${i+1}`)?.scrollIntoView({block:'nearest'});},50);
}
function delSub(i){push();subs.splice(i,1);if(activeIdx>=subs.length)activeIdx=subs.length-1;renderList();renderTL();}
function splitSub(i){
  push();
  const s=subs[i],mid=(s.start+s.end)/2,ws=s.text.split(/\s+/),h=Math.ceil(ws.length/2);
  subs.splice(i,1,{start:s.start,end:mid-.05,text:ws.slice(0,h).join(' ')},{start:mid,end:s.end,text:ws.slice(h).join(' ')});
  renderList();renderTL();
}
function mergeUp(i){
  if(!i)return;push();
  subs[i-1].end=subs[i].end;subs[i-1].text=(subs[i-1].text+' '+subs[i].text).trim();
  subs.splice(i,1);renderList();renderTL();
}

// ── Video sync ────────────────────────────────────────────────────────────
vid.addEventListener('timeupdate',()=>{
  const t=vid.currentTime,dur=vid.duration||0;
  document.getElementById('seekBar').value=dur?t/dur*100:0;
  document.getElementById('timeD').textContent=`${s2hms(t)} / ${s2hms(dur)}`;
  const idx=subs.findIndex(s=>t>=s.start&&t<s.end);
  renderOv(idx>=0?subs[idx].text:'');
  if(idx!==activeIdx){
    activeIdx=idx;
    document.querySelectorAll('.srow').forEach((r,j)=>r.classList.toggle('act',j===idx));
    if(idx>=0)document.getElementById(`row-${idx}`)?.scrollIntoView({block:'nearest',behavior:'smooth'});
  }
  if(tlVis){const px=tlT2X(t),W=tlCv.width;if(px<W*.1||px>W*.88){tlOff=t-W*.2/tlPPS;tlClamp();}}
});
vid.addEventListener('play',()=>{document.getElementById('btnPlay').textContent='⏸';tlStartLoop();});
vid.addEventListener('pause',()=>{document.getElementById('btnPlay').textContent='▶';tlStopLoop();renderTL();});
vid.addEventListener('ended',()=>{document.getElementById('btnPlay').textContent='▶';tlStopLoop();});
vid.addEventListener('loadedmetadata',initTL);

document.getElementById('seekBar').addEventListener('input',function(){vid.currentTime=vid.duration*this.value/100;renderTL();});
function togglePlay(){vid.paused?vid.play():vid.pause();}
function skip(s){vid.currentTime=Math.max(0,vid.currentTime+s);renderTL();}

document.addEventListener('keydown',e=>{
  const tag=document.activeElement.tagName,ctrl=e.ctrlKey||e.metaKey;
  if(ctrl&&e.code==='KeyF'){e.preventDefault();openSearch();return;}
  if(ctrl&&e.code==='KeyZ'&&!e.shiftKey){e.preventDefault();undo();return;}
  if(ctrl&&(e.code==='KeyY'||(e.code==='KeyZ'&&e.shiftKey))){e.preventDefault();redo();return;}
  if(tag==='INPUT'||tag==='TEXTAREA'||tag==='SELECT')return;
  if(e.code==='Space'){e.preventDefault();togglePlay();}
  if(e.code==='ArrowLeft')skip(-5);
  if(e.code==='ArrowRight')skip(5);
});

document.body.addEventListener('dragover',e=>e.preventDefault());
document.body.addEventListener('drop',e=>{e.preventDefault();const f=e.dataTransfer.files[0];if(f&&f.type.startsWith('video/'))uploadVid(f);});

// ── Upload ────────────────────────────────────────────────────────────────
function onVidPick(inp){const f=inp.files[0];if(f)uploadVid(f);}
function uploadVid(file){
  const fd=new FormData();fd.append('video',file);
  showLoading('上傳影片中…','正在上傳到本機伺服器，請稍候',false,null);
  fetch('/api/upload',{method:'POST',body:fd}).then(r=>r.json()).then(d=>{
    hideLoading();
    if(d.error){alert('上傳失敗：'+d.error);return;}
    videoId=d.video_id;loadVid(d.url);
    document.getElementById('pLbl').textContent='影片就緒';
  }).catch(e=>{hideLoading();alert('上傳失敗：'+e);});
}
function loadVid(url){
  vid.src=url;vid.style.display='block';
  document.getElementById('uHint').style.display='none';
  document.getElementById('vcArea').style.display='block';
  document.getElementById('tlSec').style.display='block';
  tlVis=true;wfPeaks=null;
  document.getElementById('btnWf').textContent='📊 波形';
  document.getElementById('btnWf').disabled=false;
  vid.load();
}

// ── SRT import ────────────────────────────────────────────────────────────
function onSrtPick(inp){
  const f=inp.files[0];if(!f)return;
  const r=new FileReader();
  r.onload=e=>{const p=parseSRT(e.target.result);if(!p.length){alert('無法解析 SRT');return;}push();subs=p;renderList();renderTL();document.getElementById('pLbl').textContent=`已載入 ${subs.length} 段`;};
  r.readAsText(f,'utf-8');inp.value='';
}
function openPaste(){document.getElementById('pasteM').classList.add('open');document.getElementById('pasteA').value='';setTimeout(()=>document.getElementById('pasteA').focus(),80);}
function closePaste(){document.getElementById('pasteM').classList.remove('open');}
function doPaste(){const t=document.getElementById('pasteA').value.trim();if(!t){closePaste();return;}const p=parseSRT(t);if(!p.length){alert('無法解析 SRT');return;}push();subs=p;renderList();renderTL();closePaste();document.getElementById('pLbl').textContent=`已載入 ${subs.length} 段`;}

// ── Shift ─────────────────────────────────────────────────────────────────
function openShift(){document.getElementById('shiftM').classList.add('open');document.getElementById('shiftV').value='0';setTimeout(()=>{document.getElementById('shiftV').focus();document.getElementById('shiftV').select();},80);}
function closeShift(){document.getElementById('shiftM').classList.remove('open');}
function doShift(){const sec=(parseFloat(document.getElementById('shiftV').value)||0)/1000;push();subs=subs.map(s=>({...s,start:Math.max(0,s.start+sec),end:Math.max(0,s.end+sec)}));renderList();renderTL();closeShift();}

// ── Search & Replace ──────────────────────────────────────────────────────
let srRes=[],srIdx=-1;
function openSearch(){document.getElementById('searchM').classList.add('open');setTimeout(()=>document.getElementById('sQ').focus(),80);}
function closeSearch(){
  document.getElementById('searchM').classList.remove('open');
  document.querySelectorAll('.shl,.scur').forEach(r=>r.classList.remove('shl','scur'));
  srRes=[];srIdx=-1;document.getElementById('sCnt').textContent='–';
}
function doSearch(){
  const q=document.getElementById('sQ').value,cs=document.getElementById('sCS').checked;
  document.querySelectorAll('.shl,.scur').forEach(r=>r.classList.remove('shl','scur'));
  srRes=[];srIdx=-1;
  if(!q){document.getElementById('sCnt').textContent='–';return;}
  subs.forEach((s,i)=>{if((cs?s.text:s.text.toLowerCase()).includes(cs?q:q.toLowerCase()))srRes.push(i);});
  document.getElementById('sCnt').textContent=srRes.length?`0/${srRes.length}`:'無結果';
  srRes.forEach(i=>document.getElementById(`row-${i}`)?.classList.add('shl'));
  if(srRes.length){srIdx=0;hlSr();}
}
function hlSr(){
  document.querySelectorAll('.scur').forEach(r=>r.classList.remove('scur'));
  if(srIdx<0||!srRes.length)return;
  document.getElementById(`row-${srRes[srIdx]}`)?.classList.add('scur');
  document.getElementById(`row-${srRes[srIdx]}`)?.scrollIntoView({block:'nearest',behavior:'smooth'});
  document.getElementById('sCnt').textContent=`${srIdx+1}/${srRes.length}`;
}
function srPrev(){if(!srRes.length)return;srIdx=(srIdx-1+srRes.length)%srRes.length;hlSr();}
function srNext(){if(!srRes.length)return;srIdx=(srIdx+1)%srRes.length;hlSr();}
function replOne(){
  if(srIdx<0||!srRes.length)return;
  const q=document.getElementById('sQ').value,r=document.getElementById('sR').value,cs=document.getElementById('sCS').checked;
  if(!q)return;push();
  const re=new RegExp(q.replace(/[.*+?^${}()|[\]\\]/g,'\\$&'),cs?'g':'gi');
  subs[srRes[srIdx]].text=subs[srRes[srIdx]].text.replace(re,r);
  renderList();doSearch();
}
function replAll(){
  const q=document.getElementById('sQ').value,r=document.getElementById('sR').value,cs=document.getElementById('sCS').checked;
  if(!q)return;push();
  const re=new RegExp(q.replace(/[.*+?^${}()|[\]\\]/g,'\\$&'),cs?'g':'gi');
  let cnt=0;subs.forEach(s=>{const b=s.text;s.text=s.text.replace(re,r);if(s.text!==b)cnt++;});
  renderList();closeSearch();alert(`已取代 ${cnt} 個字幕`);
}

// ── Export ────────────────────────────────────────────────────────────────
function toggleExp(){const m=document.getElementById('expMenu');m.classList.toggle('open');if(m.classList.contains('open'))document.addEventListener('click',expOuter,true);}
function closeExp(){document.getElementById('expMenu').classList.remove('open');document.removeEventListener('click',expOuter,true);}
function expOuter(e){if(!document.querySelector('.exp-wrap').contains(e.target))closeExp();}
function dl(content,name,mime){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([content],{type:mime+';charset=utf-8'}));a.download=name;a.click();}
function exportSRT(){
  if(!subs.length){alert('沒有字幕');return;}
  fetch('/api/export_srt',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({segments:subs})})
  .then(r=>r.blob()).then(b=>{const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='subtitles.srt';a.click();});
}
function exportVTT(){
  if(!subs.length){alert('沒有字幕');return;}
  let v='WEBVTT\n\n';
  subs.forEach((s,i)=>{v+=`${i+1}\n${s2srt(s.start).replace(',','.')} --> ${s2srt(s.end).replace(',','.')}\n${s.text}\n\n`;});
  dl(v,'subtitles.vtt','text/vtt');
}
function exportASS(){
  if(!subs.length){alert('沒有字幕');return;}
  const fn=st.fontName,fs=st.fontSize,bold=st.bold?-1:0,italic=st.italic?-1:0;
  function hex2ass(h,a=1){const r=parseInt(h.slice(1,3),16),g=parseInt(h.slice(3,5),16),b=parseInt(h.slice(5,7),16),al=Math.round((1-a)*255);return`&H${al.toString(16).padStart(2,'0').toUpperCase()}${b.toString(16).padStart(2,'0').toUpperCase()}${g.toString(16).padStart(2,'0').toUpperCase()}${r.toString(16).padStart(2,'0').toUpperCase()}`;}
  const tcol=hex2ass(st.textColor);
  const bs=st.bgStyle==='none'?1:st.bgStyle==='outline'?1:3;
  const outline=st.bgStyle==='outline'?2:0,shadow=st.bgStyle==='outline'?1:0;
  const bcol=st.bgStyle==='none'?'&H00000000':st.bgStyle==='solid'?hex2ass(st.bgColor,1):hex2ass(st.bgColor,st.bgOpacity);
  const px=Math.round(st.xPct/100*1920),py=Math.round(st.yPct/100*1080);
  const posTag=`{\\an5\\pos(${px},${py})}`;
  const hdr=`[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\nScaledBorderAndShadow: yes\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,${fn},${fs},${tcol},&H000000FF,&H00000000,${bcol},${bold},${italic},0,0,100,100,0,0,${bs},${outline},${shadow},5,10,10,0,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n`;
  const body=subs.map(s=>`Dialogue: 0,${s2ass(s.start)},${s2ass(s.end)},Default,,0,0,0,,${posTag}${s.text.replace(/\n/g,'\\N')}`).join('\n');
  dl(hdr+body+'\n','subtitles.ass','text/plain');
}

// ── Burn ──────────────────────────────────────────────────────────────────
let burnId=null,burnPoll=null;
function openBurn(){
  if(!videoId){alert('請先上傳影片');return;}
  if(!subs.length){alert('沒有字幕');return;}
  document.getElementById('burnM').classList.add('open');
  document.getElementById('btnBurn').disabled=false;
  document.getElementById('btnBurn').textContent='🎬 開始燒錄';
}
function closeBurn(){document.getElementById('burnM').classList.remove('open');clearTimeout(burnPoll);}
function startBurn(){
  const crf=parseInt(document.getElementById('burnCrf').value);
  closeBurn();
  showLoading('燒錄字幕到影片中…','正在使用 FFmpeg 處理，請稍候',true,()=>{
    if(burnId){fetch(`/api/burn_job/${burnId}/cancel`,{method:'POST'});}
    hideLoading();clearTimeout(burnPoll);burnId=null;
  });
  fetch('/api/burn_start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
    video_id:videoId,segments:subs,style:st,crf
  })}).then(r=>r.json()).then(d=>{
    if(d.error){hideLoading();alert('燒錄失敗：'+d.error);return;}
    burnId=d.burn_id;pollBurn();
  }).catch(e=>{hideLoading();alert('請求失敗：'+e);});
}
function pollBurn(){
  if(!burnId)return;
  fetch(`/api/burn_job/${burnId}`).then(r=>r.json()).then(d=>{
    updateLoading(d.progress||0,`燒錄中… ${d.progress||0}%`);
    if(d.status==='done'){hideLoading();window.location.href=`/api/burn_download/${burnId}`;burnId=null;}
    else if(d.status==='cancelled'){hideLoading();burnId=null;}
    else if(d.status==='error'){hideLoading();alert('燒錄失敗：'+(d.error||'未知錯誤'));burnId=null;}
    else burnPoll=setTimeout(pollBurn,1000);
  }).catch(()=>{burnPoll=setTimeout(pollBurn,2000);});
}

// ── Transcribe bar toggle ─────────────────────────────────────────────────
function toggleTbar(){
  const b=document.getElementById('tbarBody'),a=document.getElementById('tbarArr');
  b.classList.toggle('open');a.classList.toggle('open',b.classList.contains('open'));
}

// ── Transcribe ────────────────────────────────────────────────────────────
let txStatus='';
function startTx(){
  if(!videoId){alert('請先上傳影片');return;}
  document.getElementById('btnGo').disabled=true;
  document.getElementById('btnStop').style.display='inline-flex';
  document.getElementById('sDot').className='sdot run';
  document.getElementById('pLbl').textContent='轉錄中…';
  document.getElementById('pFill').style.width='0%';
  document.getElementById('logBox').classList.add('show');
  document.getElementById('logBox').textContent='';
  fetch('/api/transcribe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
    video_id:videoId,model:document.getElementById('mdlSel').value,
    quality:document.getElementById('qSel').value,prompt:document.getElementById('prompt').value.trim(),
    to_tc:document.getElementById('toTC').checked,delay_ms:parseInt(document.getElementById('dms').value)||0
  })}).then(r=>r.json()).then(d=>{jobId=d.job_id;pollTx();}).catch(e=>{txDone('error');document.getElementById('pLbl').textContent='請求失敗';});
}
function pollTx(){
  if(!jobId)return;
  fetch(`/api/job/${jobId}`).then(r=>r.json()).then(d=>{
    txStatus=d.status;const p=d.progress||0;
    document.getElementById('pFill').style.width=p+'%';
    document.getElementById('pLbl').textContent=
      d.status==='running'?`轉錄中… ${p}%`:d.status==='done'?`✅ 完成 ${(d.segments||[]).length} 段`:
      d.status==='cancelled'?'⚠️ 已取消':`❌ ${d.error||d.status}`;
    if(d.log?.length){document.getElementById('logBox').textContent=d.log.join('\n');document.getElementById('logBox').scrollTop=9999;}
    if(d.status==='done'&&d.segments){push();subs=d.segments;renderList();renderTL();txDone('done');}
    else if(d.status==='error'||d.status==='cancelled')txDone(d.status);
    else pollTmr=setTimeout(pollTx,1000);
  }).catch(()=>{pollTmr=setTimeout(pollTx,2000);});
}
function cancelTx(){if(jobId){fetch(`/api/job/${jobId}/cancel`,{method:'POST'});document.getElementById('pLbl').textContent='取消中…';}}
function txDone(s){document.getElementById('btnGo').disabled=false;document.getElementById('btnStop').style.display='none';document.getElementById('sDot').className='sdot '+s;clearTimeout(pollTmr);}

// ── Timeline ──────────────────────────────────────────────────────────────
const tlCv=document.getElementById('tlCv'),tlCtx=tlCv.getContext('2d');
let wfPeaks=null,tlPPS=50,tlOff=0,tlDrag=null,tlPan=null,tlRAF=null,tlVis=false,ovDrag=false,snapIndicator=null;

function tlT2X(t){return(t-tlOff)*tlPPS;}
function tlX2T(x){return x/tlPPS+tlOff;}
function tlClamp(){const mx=Math.max(0,(vid.duration||0)-tlCv.width/tlPPS);tlOff=Math.max(0,Math.min(tlOff,mx));}

function initTL(){
  const p=document.getElementById('tlSec');
  tlCv.width=p.clientWidth||p.offsetWidth;
  tlCv.height=p.clientHeight||p.offsetHeight;
  if(vid.duration){tlPPS=tlCv.width/vid.duration;tlOff=0;}
  renderTL();tlSyncZoomBar();
}
new ResizeObserver(()=>{if(tlVis)initTL();}).observe(document.getElementById('tlSec'));

function tlTickInterval(vis){for(const t of[.5,1,2,5,10,15,30,60,120,300,600,1800,3600])if(t>=vis/10)return t;return 3600;}

function renderTL(){
  const W=tlCv.width,H=tlCv.height;if(!W||!H)return;
  const ctx=tlCtx,RH=20,BT=RH+3,BH=H-RH-6;
  ctx.fillStyle='#080d14';ctx.fillRect(0,0,W,H);

  // Waveform
  if(wfPeaks&&vid.duration){
    const mid=RH+(H-RH)*.5,amp=(H-RH)*.4;
    ctx.fillStyle='#162032';
    for(let px=0;px<W;px++){
      const t=tlX2T(px);if(t<0||t>vid.duration)continue;
      const idx=Math.floor(t/vid.duration*wfPeaks.length);
      const v=wfPeaks[Math.min(idx,wfPeaks.length-1)];
      const h=Math.max(1,v*amp);
      ctx.fillRect(px,mid-h,1,h*2);
    }
  }

  // Subtitle bars — teal inactive, orange active, white border
  subs.forEach((s,i)=>{
    const x1=tlT2X(s.start),x2=tlT2X(s.end);
    if(x2<0||x1>W)return;
    const w=Math.max(3,x2-x1),act=i===activeIdx;
    const lx=Math.max(0,x1),rx=Math.min(W,x2),bw=rx-lx;
    if(bw<=0)return;

    // Fill
    ctx.fillStyle=act?'#c2410c':'#0f766e';
    ctx.globalAlpha=act?.95:.78;
    ctx.fillRect(lx,BT,bw,BH);

    // White border
    ctx.globalAlpha=act?.9:.5;
    ctx.strokeStyle=act?'#fed7aa':'#99f6e4';
    ctx.lineWidth=1.5;
    ctx.strokeRect(lx+.75,BT+.75,bw-1.5,BH-1.5);

    // Resize handles
    ctx.globalAlpha=1;
    ctx.fillStyle=act?'#fb923c':'#2dd4bf';
    if(x1>=0&&x1<=W)ctx.fillRect(x1,BT,4,BH);
    if(x2>=0&&x2<=W)ctx.fillRect(x2-4,BT,4,BH);

    // Label
    if(bw>28){
      ctx.globalAlpha=1;ctx.fillStyle='#fff';ctx.font='10px -apple-system,sans-serif';ctx.textAlign='left';
      ctx.save();ctx.beginPath();ctx.rect(lx+6,BT,bw-12,BH);ctx.clip();
      ctx.fillText(s.text.replace(/\n/g,' '),lx+6,BT+BH/2+3.5);ctx.restore();
    }
    ctx.globalAlpha=1;
  });

  // Snap indicator
  if(snapIndicator!==null){
    const sx=Math.round(tlT2X(snapIndicator));
    if(sx>=0&&sx<=W){
      ctx.save();ctx.strokeStyle='#fde047';ctx.lineWidth=1.5;ctx.setLineDash([4,3]);
      ctx.beginPath();ctx.moveTo(sx,RH+3);ctx.lineTo(sx,H);ctx.stroke();
      ctx.setLineDash([]);ctx.restore();
    }
  }

  // Ruler
  ctx.fillStyle='#0a0f1a';ctx.fillRect(0,0,W,RH);
  ctx.fillStyle='#1e2d3d';ctx.fillRect(0,RH,W,1);
  const vis=W/tlPPS,tick=tlTickInterval(vis),startT=Math.floor(tlOff/tick)*tick;
  ctx.font='9px -apple-system,sans-serif';ctx.textAlign='center';
  for(let t=startT;t<=tlOff+vis+tick;t+=tick){
    const x=Math.round(tlT2X(t));if(x<-30||x>W+30)continue;
    ctx.fillStyle='#2d4058';ctx.fillRect(x,RH-5,1,5);
    ctx.fillStyle='#4a6785';ctx.fillText(s2hms(t),x,RH-7);
  }

  // Playhead
  const px=Math.round(tlT2X(vid.currentTime||0));
  if(px>=-2&&px<=W+2){
    ctx.strokeStyle='#f85149';ctx.lineWidth=2;
    ctx.beginPath();ctx.moveTo(px,0);ctx.lineTo(px,H);ctx.stroke();
    ctx.fillStyle='#f85149';
    ctx.beginPath();ctx.moveTo(px-5,0);ctx.lineTo(px+5,0);ctx.lineTo(px,9);ctx.fill();
  }
}

function tlStartLoop(){if(tlRAF)return;function L(){renderTL();tlRAF=requestAnimationFrame(L);}tlRAF=requestAnimationFrame(L);}
function tlStopLoop(){if(tlRAF){cancelAnimationFrame(tlRAF);tlRAF=null;}}
function tlFit(){if(vid.duration){tlPPS=tlCv.width/vid.duration;tlOff=0;renderTL();tlSyncZoomBar();}}
function tlSyncZoomBar(){
  const mn=vid.duration?tlCv.width/vid.duration:1,mx=600;
  const bar=document.getElementById('tlZoomBar'),pct=document.getElementById('tlZoomPct');
  if(!bar)return;
  if(tlPPS<=mn){bar.value=0;pct.textContent='100%';return;}
  bar.value=Math.round(Math.log(tlPPS/mn)/Math.log(mx/mn)*100);
  pct.textContent=Math.round(tlPPS/mn*100)+'%';
}
function tlZoomSlide(v){
  const mn=vid.duration?tlCv.width/vid.duration:1,mx=600;
  const cx=tlCv.width/2,t=tlX2T(cx);
  tlPPS=mn*Math.pow(mx/mn,v/100);
  tlOff=t-cx/tlPPS;tlClamp();renderTL();
  document.getElementById('tlZoomPct').textContent=Math.round(tlPPS/mn*100)+'%';
}

const EDGE=7,RH_=20,SNAP_PX=10;
function snapTime(t,excludeIdx){
  let best=t,bestPixDist=SNAP_PX;
  subs.forEach((s,i)=>{
    if(i===excludeIdx)return;
    [s.start,s.end].forEach(edge=>{
      const d=Math.abs(tlT2X(edge)-tlT2X(t));
      if(d<bestPixDist){bestPixDist=d;best=edge;}
    });
  });
  snapIndicator=(best!==t)?best:null;
  return best;
}
tlCv.addEventListener('mousedown',e=>{
  if(e.button===1){e.preventDefault();tlPan={sx:e.clientX,so:tlOff};return;}
  if(e.button!==0)return;
  const rc=tlCv.getBoundingClientRect(),sc=tlCv.width/rc.width;
  const x=(e.clientX-rc.left)*sc,y=(e.clientY-rc.top)*(tlCv.height/rc.height);
  const BT=RH_+3,BH=tlCv.height-RH_-6;
  if(y<=RH_){vid.currentTime=Math.max(0,Math.min(tlX2T(x),vid.duration||0));renderTL();return;}
  for(let i=subs.length-1;i>=0;i--){
    const s=subs[i],x1=tlT2X(s.start),x2=tlT2X(s.end);
    if(y<BT||y>BT+BH||x<x1-EDGE||x>x2+EDGE)continue;
    push();
    let type='move';if(x<=x1+EDGE)type='left';else if(x>=x2-EDGE)type='right';
    tlDrag={i,type,sx:x,os:s.start,oe:s.end};
    activeIdx=i;document.querySelectorAll('.srow').forEach((r,j)=>r.classList.toggle('act',j===i));
    document.getElementById(`row-${i}`)?.scrollIntoView({block:'nearest'});
    renderTL();return;
  }
  vid.currentTime=Math.max(0,Math.min(tlX2T(x),vid.duration||0));renderTL();
});
document.addEventListener('mousemove',e=>{
  if(tlPan){const dx=e.clientX-tlPan.sx;tlOff=tlPan.so-dx/tlPPS;tlClamp();renderTL();return;}
  if(ovDrag){
    const vw=document.getElementById('vWrap'),rect=vw.getBoundingClientRect();
    const rawX=Math.max(3,Math.min(97,ovDragX0+(e.clientX-ovDragSX)/rect.width*100));
    const rawY=Math.max(3,Math.min(97,ovDragY0+(e.clientY-ovDragSY)/rect.height*100));
    const sx=_snapG(rawX,GUIDES_X,SNAP_GT);
    const sy=_snapG(rawY,GUIDES_Y,SNAP_GT);
    st.xPct=sx.val; st.yPct=sy.val;
    _showGuides(sx.snapped,sy.snapped);
    updatePosReadout(); applyOvStyle();
    return;
  }
  if(!tlDrag)return;
  const rc=tlCv.getBoundingClientRect();
  const x=(e.clientX-rc.left)*(tlCv.width/rc.width),dt=(x-tlDrag.sx)/tlPPS,s=subs[tlDrag.i],dur=tlDrag.oe-tlDrag.os;
  snapIndicator=null;
  if(tlDrag.type==='move'){s.start=snapTime(Math.max(0,tlDrag.os+dt),tlDrag.i);s.end=s.start+dur;}
  else if(tlDrag.type==='left'){s.start=snapTime(Math.max(0,Math.min(tlDrag.oe-.1,tlDrag.os+dt)),tlDrag.i);}
  else{s.end=snapTime(Math.max(s.start+.1,tlDrag.oe+dt),tlDrag.i);}
  const si=document.getElementById(`ts-${tlDrag.i}`),ei=document.getElementById(`te-${tlDrag.i}`);
  if(si)si.value=s2srt(s.start);if(ei)ei.value=s2srt(s.end);
  renderTL();
});
document.addEventListener('mouseup',()=>{
  const wasDragging=ovDrag;
  tlDrag=null;tlPan=null;ovDrag=false;
  if(wasDragging)_hideGuides();
  if(snapIndicator!==null){snapIndicator=null;renderTL();}
});

tlCv.addEventListener('mousemove',e=>{
  if(tlDrag||tlPan)return;
  const rc=tlCv.getBoundingClientRect();
  const x=(e.clientX-rc.left)*(tlCv.width/rc.width),y=(e.clientY-rc.top)*(tlCv.height/rc.height);
  const BT=RH_+3,BH=tlCv.height-RH_-6;
  if(y>BT&&y<BT+BH){
    for(let i=subs.length-1;i>=0;i--){
      const x1=tlT2X(subs[i].start),x2=tlT2X(subs[i].end);
      if(x>=x1-EDGE&&x<=x2+EDGE){tlCv.style.cursor=(x<=x1+EDGE||x>=x2-EDGE)?'ew-resize':'grab';return;}
    }
  }
  tlCv.style.cursor='crosshair';
});

// ── Waveform ──────────────────────────────────────────────────────────────
async function loadWf(){
  if(!vid.src)return;
  const btn=document.getElementById('btnWf');
  btn.textContent='⏳ 載入…';btn.disabled=true;
  try{
    const ac=new(window.AudioContext||window.webkitAudioContext)();
    const decoded=await ac.decodeAudioData(await(await fetch(vid.src)).arrayBuffer());
    ac.close();
    const data=decoded.getChannelData(0),N=Math.min(tlCv.width*4,10000),block=Math.floor(data.length/N);
    const peaks=new Float32Array(N);let mx=0;
    for(let i=0;i<N;i++){let s=0;for(let j=0;j<block;j++)s+=Math.abs(data[i*block+j]||0);peaks[i]=s/block;if(peaks[i]>mx)mx=peaks[i];}
    if(mx>0)for(let i=0;i<N;i++)peaks[i]/=mx;
    wfPeaks=peaks;btn.textContent='✅ 波形';renderTL();
  }catch(e){btn.textContent='📊 波形';btn.disabled=false;alert('無法載入波形：'+e.message);}
}

// ── Wheel zoom sync ───────────────────────────────────────────────────────
tlCv.addEventListener('wheel',e=>{
  e.preventDefault();
  const rc=tlCv.getBoundingClientRect(),x=(e.clientX-rc.left)*(tlCv.width/rc.width),t=tlX2T(x);
  const mn=vid.duration?tlCv.width/vid.duration:.1;
  tlPPS=Math.max(mn,Math.min(600,tlPPS*(e.deltaY<0?1.25:.8)));
  tlOff=t-x/tlPPS;tlClamp();renderTL();tlSyncZoomBar();
},{passive:false});

// ── Init ──────────────────────────────────────────────────────────────────
fetch('/api/models').then(r=>r.json()).then(d=>{
  document.getElementById('mdlSel').querySelectorAll('option').forEach(o=>{if(d.cached?.includes(o.value))o.textContent+=' ✅';});
}).catch(()=>{});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import webbrowser, threading
    port = 5500
    print(f"字幕編輯器啟動中… http://localhost:{port}")
    threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
