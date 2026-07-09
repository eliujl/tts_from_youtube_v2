# y2tts — YouTube / Local → Transcribe → TTS

A local pipeline to:
- **Download** YouTube videos/playlists (or use a **local video/audio file**)
- **Transcribe** with **faster-whisper**
- Optionally **polish messy transcripts with AI**, either locally through Ollama or through an explicitly approved compatible endpoint
- Optionally **re-synthesize** clean speech with **Piper** (default), **Microsoft Edge Neural TTS**, or **Coqui TTS**
- Export raw, basic-clean, optional AI-polished, subtitle, manifest, and TTS artifacts

> This project is intended to run **locally**. For best results, install it in a **dedicated virtual environment** to avoid dependency conflicts with your existing Python stack.

---

## Features

### Inputs
- **YouTube** video or **playlist**
- **Local file upload**: `.mp4/.mkv/.webm/.mov/.mp3/.wav/.m4a/.flac/.ogg/.aac`
- **Text/document input for direct TTS**: `.txt/.md/.vtt/.pdf` (skips ASR)
- PDFs must contain selectable text; scanned/image-only PDFs require OCR first

### Modes
- **Run (ASR + TTS)**: download/extract audio → transcribe → synthesize
- **Transcribe only**: download/extract audio → transcribe
- **Optional AI polish** works with either mode and writes a separate reviewable artifact
- **Download only (YouTube)**: just download media, no ASR/TTS
  - `kind=video`: best video+audio merged (mp4 default)
  - `kind=audio`: best audio extracted to wav/mp3 (`--audio-format`)

### Outputs (per item)
- `<title>.transcript.txt` — raw transcript (segment concatenation)
- `<title>.transcript_clean.txt` — basic cleanup
- `<title>.transcript_tts.txt` — optional AI-polished, TTS-ready transcript
- `<title>.transcript.srt` — subtitles (segment-level)
- `<title>.transcript.json` — segments + optional word timestamps
- `<title>.manifest.json` — run metadata
- `<title>.audio_16k.wav` — normalized ASR input audio
- `<title>.wav` (and optional `<title>.mp3`) — synthesized TTS output

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
pip install -e ".[ui,asr,tts_piper,tts_microsoft]"
```

### macOS / Linux
```bash
cd path/to/repo

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip setuptools wheel
pip install "numpy<2"
pip install -e ".[ui,asr,tts_piper,tts_microsoft]"
```

For the online Microsoft neural voices used by OpenClaw's Microsoft provider:

```bash
pip install -e ".[tts_microsoft]"
```

---

## Quickstart (CLI)

### 1) End-to-end: YouTube → ASR → TTS
```bash
y2tts run "https://youtu.be/VIDEO_ID" --out out --mp3 \
  --model distil-large-v3 --vad \
  --tts piper --piper-voice en_US-lessac-medium \
  --tts-speed 0.9 \
  --preserve-paragraph-breaks
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
# slower speech
y2tts local "C:\path\to\video.mp4" --out out --model distil-large-v3 --tts piper --tts-speed 0.85
```

### 5) Text file to speech (no ASR)
```bash
y2tts local "C:\path\to\script.txt" --out out --tts piper --mp3
y2tts local "C:\path\to\notes.md" --out out --tts piper
y2tts local "C:\path\to\captions.vtt" --out out --tts piper
y2tts local "C:\path\to\document.pdf" --out out --tts microsoft --mp3
```

### 6) Offline AI polish with Ollama

Install and run [Ollama](https://docs.ollama.com/), then install a text model of your choice. The model must already exist locally before running the pipeline.

Polish only, so the result can be reviewed before TTS:

```bash
y2tts local "C:\path\to\transcript.txt" --out out --tts none \
  --polish ollama --polish-model llama3.1:8b \
  --preserve-paragraph-breaks
```

Fully offline polish and TTS in one run:

```bash
y2tts local "C:\path\to\transcript.txt" --out out --mp3 \
  --polish ollama --polish-model llama3.1:8b \
  --polish-glossary "C:\path\to\glossary.txt" \
  --tts piper --piper-voice en_US-ryan-high --piper-data-dir voices \
  --preserve-paragraph-breaks
```

Ollama defaults to its native API at `http://127.0.0.1:11434`. The tool rejects a non-loopback URL when `--polish ollama` is selected.

On systems with an older or undersized GPU, force reliable CPU execution:

```bash
y2tts local transcript.txt --tts none --polish ollama \
  --polish-model YOUR_INSTALLED_MODEL --ollama-num-gpu 0
```

The WebUI defaults Ollama GPU layers to `0` for stability. Set it to `-1` to let Ollama choose automatically on newer hardware.
The WebUI also detects installed Ollama generation models and preselects a practical one; embedding models are excluded.

Polishing is intentionally conservative:

- numeric claims must remain unchanged
- suspicious summarization or expansion is rejected
- weak models may fail validation instead of producing a fluent but inaccurate transcript
- successful chunks are checkpointed in `<output>/.polish_chunks/`, so a failed long run can resume
- a larger instruction-following model is strongly recommended for lectures and technical material

### 7) Other OpenAI-compatible polish endpoints

Compatible local servers such as LM Studio or llama.cpp can use the same endpoint shape. A non-local URL is blocked unless online polishing is explicitly allowed:

```bash
y2tts local "C:\path\to\transcript.txt" --out out --tts none \
  --polish openai-compatible --polish-model MODEL_NAME \
  --polish-base-url https://provider.example/v1 \
  --polish-api-key-env PROVIDER_API_KEY \
  --allow-online-polish
```

`--allow-online-polish` means transcript chunks may leave the computer. API keys are read from environment variables and should not be placed on the command line or in the UI.

### 8) Microsoft Edge Neural TTS (online, no API key)
```bash
y2tts local "C:\path\to\script.txt" --out out --tts microsoft --mp3 \
  --microsoft-voice en-US-MichelleNeural --tts-speed 0.9 \
  --preserve-paragraph-breaks
```

Microsoft TTS sends transcript text to Microsoft's online Edge speech service. It is best-effort and requires internet access.
The WebUI offers `en-US-MichelleNeural` and `zh-CN-XiaoxiaoNeural` by default and also accepts any other Microsoft Edge neural voice name.

### Recommended high-quality transcript workflow

1. Keep the original ASR transcript unchanged.
2. Run AI polish with `--tts none`; this creates `<title>.transcript_tts.txt` without overwriting the raw or basic-clean transcript.
3. Review and edit `<title>.transcript_tts.txt`.
4. Run the reviewed file through the chosen TTS backend:

```bash
y2tts local "C:\path\to\transcript_tts.txt" --out out --tts microsoft --mp3 \
  --microsoft-voice en-US-MichelleNeural --tts-speed 0.9 \
  --preserve-paragraph-breaks
```

For one-click operation, enable polishing and TTS in the same command. Use the two-pass flow when fidelity matters. Use Ollama plus Piper when all text must remain offline. Microsoft Neural usually sounds more natural, but sends the reviewed text to Microsoft's online speech service.

---

## Web UI (Gradio)

Install UI extra:
```bash
pip install -e ".[ui,asr,tts_piper,tts_microsoft]"
```

Launch:
```bash
y2tts ui --host 127.0.0.1 --port 7860
```

Open:
- http://127.0.0.1:7860

UI supports:
- **YouTube** URL or **Local file** upload
- Local file accepts one or many audio/video/text files
- `.txt/.md/.vtt/.pdf` local inputs go directly to optional polish/TTS
- PDF extraction supports selectable-text PDFs; scanned/image-only PDFs require OCR first
- Task selection: **Run / Transcribe only / Download only (YouTube)**
- Multiple local uploads are processed one by one into separate output folders
- A visible **Cancel** button for queued or in-flight UI jobs
- ASR controls: model/device/VAD/language/word timestamps
- Optional AI polish controls: none/Ollama/OpenAI-compatible, model, endpoint, glossary, chunk size, timeout, and online consent
- Optional custom polish-requirements upload (`.txt`/`.md`) for document-specific audio editing rules
- TTS controls: Piper/Microsoft/Coqui/none, voice selection, mp3 output
- WebUI defaults to `TTS backend = none` and `Also output mp3 = on`
- TTS speed control (`1.0` normal, `<1` slower, `>1` faster)
- Optional paragraph-preserving cleanup for more natural TTS pauses (`--preserve-paragraph-breaks`)
- Preview prefers the AI-polished transcript when one was generated

Stop the local Web UI server later with:
```bash
y2tts ui-stop
```

Check whether this workspace has a recorded UI server with:
```bash
y2tts ui-status
```

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
