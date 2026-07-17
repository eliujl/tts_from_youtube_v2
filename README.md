# y2tts — YouTube / Webpage / Local → Transcribe → TTS

A local pipeline to:
- **Download** YouTube videos/playlists (or use a **local video/audio file**)
- **Extract readable article text** from an ordinary webpage URL
- **Transcribe** with **faster-whisper**
- Optionally **polish messy transcripts with AI**, either locally through Ollama or through an explicitly approved compatible endpoint
- Optionally **re-synthesize** clean speech with **Piper** (default), **Microsoft Edge Neural TTS**, or **Coqui TTS**
- Export raw, basic-clean, optional AI-polished, subtitle, manifest, and TTS artifacts

> This project is intended to run **locally**. For best results, install it in a **dedicated virtual environment** to avoid dependency conflicts with your existing Python stack.

---

## Features

### Inputs
- **YouTube** video or **playlist**
- **Webpage article URL**: ordinary `http://` or `https://` pages with readable HTML article text
- **Local file upload**: `.mp4/.mkv/.webm/.mov/.mp3/.wav/.m4a/.flac/.ogg/.aac`
- **Text/document input for direct TTS**: `.txt/.md/.vtt/.pdf` (skips ASR)
- PDFs must contain selectable text; scanned/image-only PDFs require OCR first

### Modes
- **Run (ASR + TTS)**: download/extract audio → transcribe → synthesize
- **Webpage/document to TTS**: extract/load text → cleanup → optional polish → optional synthesize; skips ASR
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

For webpage runs, the manifest records `input_kind=webpage`, the original `source_url`, and the redirected `source_final_url`. The fetched HTML is not saved; the extracted raw text is the reviewable source artifact.

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

### 6) Webpage article to text / speech (no ASR)
```powershell
# Extract only, then review transcript_clean.txt
y2tts webpage "https://example.org/article" --out out --tts none

# Extract and polish locally, then review transcript_tts.txt
y2tts webpage "https://example.org/article" --out out --tts none `
  --polish ollama --polish-model MODEL_NAME

# After review, synthesize (this fetches/extracts the page again)
y2tts webpage "https://example.org/article" --out out --tts microsoft --mp3 `
  --microsoft-voice en-US-MichelleNeural

# Logged-in pages such as Reddit can use Firefox browser cookies
y2tts webpage "https://www.reddit.com/r/example/comments/POST/thread/" --out out --tts none `
  --web-cookies-from-browser firefox

# Or use an exported Netscape-format cookies.txt file
y2tts webpage "https://www.reddit.com/r/example/comments/POST/thread/" --out out --tts none `
  --web-cookies-file cookies.txt
```

Webpage mode fetches the page over the network, extracts main readable text, and skips ASR. Paywalls, login pages, bot protection, JavaScript-only rendering, and unusually structured pages may fail; save or upload the article as `.txt` when extraction is blocked.
If a page needs your existing browser login, add `--web-cookies-from-browser firefox`, `chrome`, `edge`, or another supported browser. Use `--web-browser-profile` when the login is in a non-default profile. You normally do not need to keep Firefox open; staying logged in is the important part, and closing Firefox can avoid cookie-database lock errors.

Reddit post URLs use Reddit's thread response and include the original post plus nested comments and replies. If Reddit omits a large "more comments" group from one response, the transcript reports the omitted count instead of silently dropping it.

On Windows, current Brave/Chrome releases may report `Failed to decrypt with DPAPI` because their cookies use app-bound encryption. A profile path, closing Brave/Chrome, or running as administrator usually will not fix that. In that case, export the needed cookies in Netscape `cookies.txt` format and use `--web-cookies-file cookies.txt`, or log into the page with Firefox and choose Firefox cookies. The Web UI provides an **Exported cookies.txt** upload under **Webpage Settings**.

### 7) Offline AI polish with Ollama

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

### 8) Other OpenAI-compatible polish endpoints

Compatible local servers such as LM Studio or llama.cpp can use the same endpoint shape. A non-local URL is blocked unless online polishing is explicitly allowed:

```bash
y2tts local "C:\path\to\transcript.txt" --out out --tts none \
  --polish openai-compatible --polish-model MODEL_NAME \
  --polish-base-url https://provider.example/v1 \
  --polish-api-key-env PROVIDER_API_KEY \
  --allow-online-polish
```

`--allow-online-polish` means transcript chunks may leave the computer. API keys are read from environment variables and should not be placed on the command line or in the UI.

### 9) Microsoft Edge Neural TTS (online, no API key)
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
- **YouTube** URL, **Webpage** article URL, or **Local file** upload
- Local file accepts one or many audio/video/text files
- Webpage input extracts article text and skips ASR
- Webpage input can optionally load browser cookies for logged-in pages
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

Webpage fetching is done by this local program. It may access the network URL you provide; this reveals your IP address and the requested URL to the destination site. The app blocks private/local IP literals and can use browser cookies or an exported `cookies.txt` file for logged-in pages, but it does not run page JavaScript.

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

### Webpage limitations and privacy
- Webpage extraction works best on article-like HTML with selectable text.
- Paywalls, login-only pages, bot protection, JavaScript-only pages, and unusual layouts may fail or extract too little text.
- For logged-in pages, use `--web-cookies-from-browser firefox`, another supported browser, or `--web-cookies-file cookies.txt`.
- You normally do not need to keep Firefox open for cookie use; if cookie loading fails with a lock/permission error, close Firefox completely and retry.
- On Windows, Brave/Chrome cookie loading may fail with `Failed to decrypt with DPAPI`; use Firefox cookies or an exported Netscape-format `cookies.txt` file instead.
- If extraction fails, save the article text locally and use `y2tts local article.txt ...`.
- Fetching a webpage reveals your IP address and requested URL to the site.
- Remote AI polishing sends extracted text to the configured service when `--allow-online-polish` is enabled.
- Microsoft TTS sends extracted text to Microsoft's online Edge speech service.
- Ollama polishing plus Piper TTS keeps post-fetch processing local.

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
