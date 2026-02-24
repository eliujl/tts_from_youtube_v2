# y2tts — YouTube / Local → Transcribe → TTS

A local pipeline to:
- **Download** YouTube videos/playlists (or use a **local video/audio file**)
- **Transcribe** with **faster-whisper**
- Optionally **re-synthesize** clean speech with **Piper** (default) or **Coqui TTS**
- Export artifacts: `transcript.txt`, `transcript_clean.txt`, `transcript.srt`, `transcript.json`, `manifest.json`, and `tts.wav` / `tts.mp3`

> This project is intended to run **locally**. For best results, install it in a **dedicated virtual environment** to avoid dependency conflicts with your existing Python stack.

---

## Features

### Inputs
- **YouTube** video or **playlist**
- **Local file upload**: `.mp4/.mkv/.webm/.mov/.mp3/.wav/.m4a/.flac/.ogg/.aac`
- **Text input for direct TTS**: `.txt/.vtt` (skips ASR)

### Modes
- **Run (ASR + TTS)**: download/extract audio → transcribe → synthesize
- **Transcribe only**: download/extract audio → transcribe
- **Download only (YouTube)**: just download media, no ASR/TTS
  - `kind=video`: best video+audio merged (mp4 default)
  - `kind=audio`: best audio extracted to wav/mp3 (`--audio-format`)

### Outputs (per item)
- `transcript.txt` — raw transcript (segment concatenation)
- `transcript_clean.txt` — basic cleanup
- `transcript.srt` — subtitles (segment-level)
- `transcript.json` — segments + optional word timestamps
- `manifest.json` — run metadata
- `tts.wav` (and optional `tts.mp3`)

---

## Requirements

- Python **3.10+** (3.11 recommended)
- **ffmpeg** installed and available in `PATH`

Check ffmpeg:
```bash
ffmpeg -version
```

---

## Install (recommended: virtual environment)

### Windows PowerShell
```powershell
cd path\to\repo

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -U pip setuptools wheel

# Recommended for compatibility with many existing scientific stacks
pip install "numpy<2"

# Install with ASR + Piper + Web UI
pip install -e ".[ui,asr,tts_piper]"
```

### macOS / Linux
```bash
cd path/to/repo

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip setuptools wheel
pip install "numpy<2"
pip install -e ".[ui,asr,tts_piper]"
```

---

## Quickstart (CLI)

### 1) End-to-end: YouTube → ASR → TTS
```bash
y2tts run "https://youtu.be/VIDEO_ID" --out out --mp3 \
  --model distil-large-v3 --vad \
  --tts piper --piper-voice en_US-lessac-medium
```

### 2) Transcribe only (YouTube or playlist)
```bash
y2tts transcribe "https://youtube.com/playlist?list=..." --out out --model distil-large-v3
```

### 3) Download only (YouTube)
```bash
# Download merged video
y2tts download "https://youtube.com/playlist?list=..." --out downloads --kind video

# Download audio-only (wav)
y2tts download "https://youtu.be/VIDEO_ID" --out downloads --kind audio

# Download audio-only (mp3)
y2tts download "https://youtu.be/VIDEO_ID" --out downloads --kind audio --audio-format mp3
```

### 4) Local file (no YouTube required)
```bash
y2tts local "C:\path\to\video.mp4" --out out --model distil-large-v3 --tts piper --mp3
```

### 5) Text file to speech (no ASR)
```bash
y2tts local "C:\path\to\script.txt" --out out --tts piper --mp3
y2tts local "C:\path\to\captions.vtt" --out out --tts piper
```

---

## Web UI (Gradio)

Install UI extra:
```bash
pip install -e ".[ui,asr,tts_piper]"
```

Launch:
```bash
y2tts ui --host 127.0.0.1 --port 7860
```

Open:
- http://127.0.0.1:7860

UI supports:
- **YouTube** URL or **Local file** upload
- Local file accepts audio/video and `.txt/.vtt` for direct TTS
- Task selection: **Run / Transcribe only / Download only (YouTube)**
- ASR controls: model/device/VAD/language/word timestamps
- TTS controls: Piper/Coqui/none, mp3 output

---

## Notes / Tips

### Model selection (faster-whisper)
Common choices:
- `distil-large-v3` (fast, strong quality)
- `large-v3` (best quality, slower)
- `medium` / `small` (faster on CPU)

### ffmpeg is required
- YouTube video merging and audio extraction rely on ffmpeg.
- mp3 output also uses ffmpeg (`libmp3lame`).

### Dependency conflicts
If you have a large existing Python environment (spacy/langchain/matplotlib/etc.), install this project in its own venv. You can verify environment health with:
```bash
pip check
```

---

## License

MIT (project code).  
Note: ASR/TTS models and voices may have their own licenses—check upstream model/voice terms.
