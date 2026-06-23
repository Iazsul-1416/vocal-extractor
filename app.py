import os, sys, uuid, subprocess, traceback
from datetime import datetime

import torch
import gradio as gr
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import yt_dlp

BASE_DIR      = "/tmp/audio_workspace"
DOWNLOAD_DIR  = f"{BASE_DIR}/downloads"
SEPARATED_DIR = f"{BASE_DIR}/separated"
OUTPUT_DIR    = f"{BASE_DIR}/outputs"
for d in [BASE_DIR, DOWNLOAD_DIR, SEPARATED_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

DEFAULT_SILENCE_THRESH_DB  = -40
DEFAULT_MIN_SILENCE_LEN_MS = 500
DEFAULT_KEEP_SILENCE_MS    = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Device: {DEVICE.upper()}")


def sanitize_filename(name):
    return "".join(c if (c.isalnum() or c in "-_. ") else "_" for c in name).strip()[:80] or "untitled"


def download_youtube_audio(url):
    sid = uuid.uuid4().hex[:8]
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(DOWNLOAD_DIR, f"{sid}.%(ext)s"),
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '320'}],
        'quiet': True, 'no_warnings': True, 'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get('title', 'Untitled')
    wav = os.path.join(DOWNLOAD_DIR, f"{sid}.wav")
    if not os.path.exists(wav):
        for f in os.listdir(DOWNLOAD_DIR):
            if f.startswith(sid):
                wav = os.path.join(DOWNLOAD_DIR, f); break
    return wav, title, sid


def separate_with_demucs(audio_path, sid):
    out_dir = os.path.join(SEPARATED_DIR, sid)
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, "-m", "demucs.separate", "--two-stems=vocals",
           "-n", "htdemucs", "-o", out_dir, "-d", DEVICE, audio_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Demucs error:\n{proc.stderr[-600:]}")
    base = os.path.splitext(os.path.basename(audio_path))[0]
    folder = os.path.join(out_dir, "htdemucs", base)
    vocals = os.path.join(folder, "vocals.wav")
    if not os.path.exists(vocals):
        raise RuntimeError(f"Output not found: {folder}")
    return vocals


def remove_silence(input_path, output_path, thresh_db, min_len_ms, keep_ms):
    audio = AudioSegment.from_file(input_path)
    nonsilent = detect_nonsilent(audio, min_silence_len=int(min_len_ms), silence_thresh=int(thresh_db))
    if not nonsilent:
        audio.export(output_path, format="wav")
        return len(audio)/1000.0, len(audio)/1000.0
    result = AudioSegment.empty()
    pad = int(keep_ms)
    for s, e in nonsilent:
        result += audio[max(0, s-pad):min(len(audio), e+pad)]
    result.export(output_path, format="wav")
    return len(audio)/1000.0, len(result)/1000.0


def process_pipeline(url, thresh, min_len, keep, history, progress=gr.Progress()):
    if not url or not url.strip():
        return None, "⚠️ ضع رابط يوتيوب صحيحاً.", history
    try:
        progress(0.05, desc="📥 جاري التحميل...")
        audio_path, title, sid = download_youtube_audio(url.strip())
        progress(0.30, desc="🎼 فصل الغناء (Demucs)...")
        vocals_raw = separate_with_demucs(audio_path, sid)
        progress(0.85, desc="✂️ إزالة الصمت...")
        safe = sanitize_filename(title)
        clean_vocals = os.path.join(OUTPUT_DIR, f"{sid}_{safe}_VOCALS.wav")
        orig, new = remove_silence(vocals_raw, clean_vocals, thresh, min_len, keep)
        progress(1.0, desc="✅ اكتمل!")
        entry = {
            "title": title, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "vocals_path": clean_vocals,
            "orig_dur": round(orig, 1), "new_dur": round(new, 1),
        }
        history = [entry] + (history or [])
        status = (f"✅ تمت المعالجة بنجاح\n🎵 {title}\n"
                  f"⏱ {round(orig,1)}s → {round(new,1)}s  (اختُصر {round(orig-new,1)}s)")
        return clean_vocals, status, history
    except Exception as e:
        return None, f"❌ خطأ:\n{str(e)}\n\n{traceback.format_exc()[-400:]}", history


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap');
html, body, .gradio-container, .gradio-container > .main, .app, .wrap, .contain, .gradio-container .gap {
  background: #ffffff !important;
}
.gradio-container { max-width: 720px !important; margin: 0 auto !important; direction: rtl; min-height: 100vh; }
.gradio-container, .gradio-container * { font-family: 'IBM Plex Sans Arabic', -apple-system, sans-serif !important; }
footer { display: none !important; }
.app-head { text-align: center; padding: 36px 0 14px; }
.app-head h1 {
  font-size: 32px; font-weight: 700; margin: 0 0 8px;
  background: linear-gradient(120deg, #0284c7 0%, #2563eb 50%, #3b82f6 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.5px;
}
.app-head p { color: #64748b !important; font-size: 14px; margin: 0; font-weight: 400; }
.gradio-container .block, .gradio-container .form, .gradio-container .gr-group, .gradio-container .panel {
  background: transparent !important; border: none !important; box-shadow: none !important;
}
.gradio-container, .gradio-container *, .gradio-container .gr-markdown, .gradio-container .gr-markdown *,
.gradio-container p, .gradio-container h1, .gradio-container h2, .gradio-container h3, .gradio-container h4 {
  color: #1e293b;
}
label span { color: #334155 !important; font-weight: 600 !important; font-size: 14px !important; }
.gradio-container textarea, .gradio-container input[type="text"] {
  background: #f8fafc !important; border: 1px solid #cbd5e1 !important; border-radius: 16px !important;
  color: #1e293b !important; font-size: 15px !important; padding: 16px !important; transition: all .25s ease;
}
.gradio-container textarea:focus, .gradio-container input[type="text"]:focus {
  border-color: #3b82f6 !important; background: #fff !important; box-shadow: 0 0 0 4px rgba(59,130,246,0.12) !important;
}
.gradio-container textarea::placeholder { color: #94a3b8 !important; }
.gradio-container button.primary, .gradio-container .gr-button-primary {
  background: linear-gradient(120deg, #0284c7, #2563eb) !important; border: none !important; border-radius: 16px !important;
  color: #fff !important; font-weight: 600 !important; font-size: 16px !important; padding: 15px !important;
  box-shadow: 0 8px 24px -8px rgba(37,99,235,0.45) !important; transition: transform .2s ease, box-shadow .2s ease;
}
.gradio-container button.primary:hover { transform: translateY(-2px); box-shadow: 0 14px 34px -10px rgba(37,99,235,0.55) !important; }
.gradio-container button.primary * { color: #fff !important; }
.gradio-container button.secondary {
  background: #f1f5f9 !important; border: 1px solid #cbd5e1 !important; color: #334155 !important; border-radius: 12px !important;
}
.gradio-container button.secondary * { color: #334155 !important; }
.tab-nav { border-bottom: 1px solid #e2e8f0 !important; gap: 6px; justify-content: center; margin-bottom: 10px; }
.tab-nav button {
  background: transparent !important; border: none !important; color: #94a3b8 !important; font-weight: 500 !important;
  font-size: 15px !important; padding: 10px 18px !important; border-radius: 12px 12px 0 0 !important;
}
.tab-nav button.selected { background: #eff6ff !important; color: #2563eb !important; }
.gradio-container audio { width: 100% !important; height: 52px !important; margin-top: 6px; }
.gradio-container .audio-container, .gradio-container [data-testid="waveform"] {
  background: #f8fafc !important; border: 1px solid #e2e8f0 !important; border-radius: 16px !important; padding: 8px !important;
}
.history-item {
  background: #f8fafc !important; border: 1px solid #e2e8f0 !important; border-radius: 18px !important;
  padding: 18px !important; margin-bottom: 14px !important;
}
.history-item *, .history-item h4 { color: #1e293b !important; }
"""

with gr.Blocks(theme=gr.themes.Base(primary_hue="blue", secondary_hue="sky", neutral_hue="slate"),
               css=CUSTOM_CSS, title="Vocal Extractor Pro") as demo:
    history_state = gr.State([])
    gr.HTML("""
    <div class="app-head">
      <h1>🎙️ Vocal Extractor Pro</h1>
      <p>استخراج الغناء النقي من يوتيوب — بواسطة Meta Demucs</p>
    </div>
    """)
    with gr.Tabs():
        with gr.Tab("🎬 معالجة جديدة"):
            youtube_input = gr.Textbox(label="رابط الفيديو من يوتيوب",
                                       placeholder="https://www.youtube.com/watch?v=...", lines=1)
            with gr.Accordion("⚙️ إعدادات إزالة الصمت", open=False):
                thresh_s = gr.Slider(-70, -20, value=DEFAULT_SILENCE_THRESH_DB, step=1, label="حد الصمت (dB)")
                minlen_s = gr.Slider(100, 2000, value=DEFAULT_MIN_SILENCE_LEN_MS, step=50, label="أقل مدة للصمت (ms)")
                keep_s   = gr.Slider(0, 500, value=DEFAULT_KEEP_SILENCE_MS, step=10, label="هامش الصمت المُبقى (ms)")
            process_btn = gr.Button("🚀 ابدأ المعالجة", variant="primary", size="lg")
            status_box  = gr.Textbox(label="الحالة", lines=4, interactive=False)
            vocals_output = gr.Audio(label="🎤 الغناء المصفى", type="filepath")
        with gr.Tab("📚 السجل والمحفوظات"):
            gr.Markdown("### كل العمليات السابقة في هذه الجلسة")
            clear_btn = gr.Button("🗑️ مسح السجل", variant="secondary", size="sm")
            @gr.render(inputs=history_state)
            def render_history(history):
                if not history:
                    gr.Markdown("📭 السجل فارغ. ابدأ بمعالجة فيديو.")
                    return
                for idx, item in enumerate(history, start=1):
                    with gr.Group(elem_classes="history-item"):
                        gr.Markdown(f"#### {idx}. 🎵 {item['title']}\n"
                                    f"🕐 {item['timestamp']}  •  ⏱ {item['new_dur']}s (من {item['orig_dur']}s)")
                        gr.Audio(value=item['vocals_path'], label="🎤 الغناء المصفى", type="filepath")
    process_btn.click(process_pipeline,
                      inputs=[youtube_input, thresh_s, minlen_s, keep_s, history_state],
                      outputs=[vocals_output, status_box, history_state])
    clear_btn.click(lambda: [], None, [history_state])

demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=8000, show_api=False)
