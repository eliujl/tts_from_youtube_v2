from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr

from .pipeline import RunConfig, download_only, run_local_file, run_many
from .polish import list_ollama_models


def _coerce_file_path(v: Any) -> str | None:
    """Gradio 4/5 may return str, dict, or FileData-like objects for File components."""
    if v is None:
        return None
    if isinstance(v, str):
        return v
    # Gradio 5: dict payloads are common
    if isinstance(v, dict):
        # Common keys across versions
        for k in ("path", "name", "file", "tmp_path", "tempfile"):
            p = v.get(k)
            if isinstance(p, str):
                return p
        return None
    # FileData-like: has .path
    p = getattr(v, "path", None)
    if isinstance(p, str):
        return p
    return None


def _coerce_dl_kind(v: Any) -> str:
    """Dropdown should be 'video'/'audio', but some Gradio versions can pass non-string payloads."""
    if isinstance(v, str) and v in ("video", "audio"):
        return v
    # If some unexpected payload arrives, default safely.
    return "video"


def _coerce_dl_audio_format(v: Any) -> str:
    """Dropdown should be 'wav'/'mp3', but some Gradio versions can pass non-string payloads."""
    if isinstance(v, str) and v in ("wav", "mp3"):
        return v
    return "wav"


def _coerce_source_type(v: Any) -> str:
    """Normalize source selector payload across Gradio versions."""
    if isinstance(v, str) and v in ("YouTube", "Local file"):
        return v
    if isinstance(v, dict):
        for k in ("value", "label", "name"):
            sv = v.get(k)
            if isinstance(sv, str) and sv in ("YouTube", "Local file"):
                return sv
    return "YouTube"


def _coerce_task(v: Any) -> str:
    """Normalize task selector payload across Gradio versions."""
    valid = ("Run (ASR + TTS)", "Transcribe only", "Download only (YouTube)")
    if isinstance(v, str) and v in valid:
        return v
    if isinstance(v, dict):
        for k in ("value", "label", "name"):
            tv = v.get(k)
            if isinstance(tv, str) and tv in valid:
                return tv
    return "Run (ASR + TTS)"


def _collect_artifacts(out_dirs: list[Path]) -> tuple[str, str, list[str]]:
    """
    Return: (status_text, transcript_preview, file_paths_for_download)
    Includes transcripts/TTS artifacts, manifests, and (for download-only) media files.
    """
    files: list[Path] = []
    summary_lines: list[str] = []

    # Standard artifacts (legacy naming)
    std_names = [
        "transcript.txt",
        "transcript_clean.txt",
        "transcript.srt",
        "transcript.json",
        "manifest.json",
    ]

    media_exts = {".mp4", ".mkv", ".webm", ".mov", ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}

    for d in out_dirs:
        summary_lines.append(str(d))

        # Legacy standard files (if present)
        for name in std_names:
            p = d / name
            if p.exists():
                files.append(p)

        # New/legacy manifest(s) drive artifact discovery.
        manifest_paths: list[Path] = []
        legacy_manifest = d / "manifest.json"
        if legacy_manifest.exists():
            manifest_paths.append(legacy_manifest)
        manifest_paths.extend(sorted(d.glob("*.manifest.json")))

        for m in manifest_paths:
            files.append(m)
            try:
                payload = json.loads(m.read_text(encoding="utf-8"))
                artifacts = payload.get("artifacts", {})
                if isinstance(artifacts, dict):
                    for v in artifacts.values():
                        if isinstance(v, str) and v.strip():
                            p = Path(v)
                            if p.exists():
                                files.append(p)
                tts_out = payload.get("tts", {}).get("output")
                if isinstance(tts_out, str) and tts_out.strip():
                    p = Path(tts_out)
                    if p.exists():
                        files.append(p)
            except Exception:
                pass

        # If this looks like download-only, include media files in the folder too
        for p in sorted(d.glob("*")):
            if p.is_file() and p.suffix.lower() in media_exts:
                files.append(p)

    # Deduplicate preserving order
    seen = set()
    uniq: list[Path] = []
    for f in files:
        if f not in seen:
            uniq.append(f)
            seen.add(f)

    # Preview the polished TTS transcript when available, otherwise basic-clean text.
    preview = ""
    for d in reversed(out_dirs):
        candidates: list[Path] = []
        candidates.append(d / "transcript_tts.txt")  # legacy
        candidates.extend(sorted(d.glob("*.transcript_tts.txt")))
        candidates.append(d / "transcript_clean.txt")  # legacy
        candidates.extend(sorted(d.glob("*.transcript_clean.txt")))
        manifests = []
        if (d / "manifest.json").exists():
            manifests.append(d / "manifest.json")
        manifests.extend(sorted(d.glob("*.manifest.json")))
        for m in manifests:
            try:
                payload = json.loads(m.read_text(encoding="utf-8"))
                artifacts = payload.get("artifacts", {})
                clean_str = artifacts.get("transcript_clean")
                polish_str = artifacts.get("transcript_tts")
                if isinstance(clean_str, str) and clean_str.strip():
                    candidates.insert(0, Path(clean_str))
                if isinstance(polish_str, str) and polish_str.strip():
                    candidates.insert(0, Path(polish_str))
            except Exception:
                pass

        for p in candidates:
            if p.exists():
                try:
                    preview = p.read_text(encoding="utf-8")
                except Exception:
                    preview = ""
                if preview:
                    break
        if preview:
            break

    return "\n".join(summary_lines), preview, [str(p) for p in uniq]


def _run(
    source_type: str,
    youtube_url: str,
    local_file: Any,
    out_dir: str,
    task: str,
    # ASR
    model: str,
    device: str,
    compute_type: str,
    lang: str,
    vad: bool,
    word_ts: bool,
    # AI polish
    polish_backend: str,
    polish_model: str,
    polish_base_url: str,
    polish_api_key_env: str,
    polish_glossary: Any,
    polish_chunk_chars: int,
    polish_timeout: int,
    ollama_num_gpu: int,
    allow_online_polish: bool,
    # TTS
    tts: str,
    tts_speed: float,
    preserve_paragraph_breaks: bool,
    mp3: bool,
    piper_voice: str,
    piper_data_dir: str,
    piper_cuda: bool,
    microsoft_voice: str,
    coqui_model: str,
    coqui_speaker_wav: Any,
    coqui_language: str,
    # Download-only options
    dl_kind: Any,
    dl_audio_format: Any,
    progress: gr.Progress | None = None,
):
    if progress is None:
        progress = gr.Progress(track_tqdm=True)

    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    source_type_norm = _coerce_source_type(source_type)
    task_norm = _coerce_task(task)
    lf = _coerce_file_path(local_file)
    dl_kind_norm = _coerce_dl_kind(dl_kind)
    dl_audio_format_norm = _coerce_dl_audio_format(dl_audio_format)

    # Download-only is YouTube-only
    if task_norm.startswith("Download only"):
        if source_type_norm != "YouTube":
            raise gr.Error("Download-only is only available for YouTube sources.")
        if not youtube_url or not youtube_url.strip():
            raise gr.Error("Please provide a YouTube video or playlist URL.")
        cfg = RunConfig(out_dir=out_path, tts_backend="none", make_mp3=False)
        progress(0.05, desc="Downloading…")
        out_dirs = download_only(
            youtube_url.strip(),
            cfg,
            kind=dl_kind_norm,
            audio_format=dl_audio_format_norm,
        )
        return _collect_artifacts(out_dirs)

    # ASR/TTS runs
    cfg = RunConfig(
        out_dir=out_path,
        asr_backend="faster-whisper",
        asr_model=model,
        asr_device=device,
        asr_compute_type=compute_type.strip() or None,
        asr_language=lang.strip() or None,
        asr_vad=vad,
        asr_word_timestamps=word_ts,
        polish_backend=polish_backend,  # type: ignore[arg-type]
        polish_model=(polish_model or "").strip(),
        polish_base_url=polish_base_url.strip(),
        polish_api_key_env=polish_api_key_env.strip(),
        polish_glossary_path=(
            Path(_coerce_file_path(polish_glossary)).expanduser().resolve()
            if _coerce_file_path(polish_glossary)
            else None
        ),
        polish_chunk_chars=int(polish_chunk_chars),
        polish_timeout_seconds=int(polish_timeout),
        polish_ollama_num_gpu=(int(ollama_num_gpu) if int(ollama_num_gpu) >= 0 else None),
        polish_allow_remote=allow_online_polish,
        tts_backend=tts,  # type: ignore[arg-type]
        tts_speed=tts_speed,
        preserve_paragraph_breaks=preserve_paragraph_breaks,
        make_mp3=mp3,
        piper_voice=piper_voice,
        piper_data_dir=Path(piper_data_dir).expanduser().resolve() if piper_data_dir.strip() else None,
        piper_use_cuda=piper_cuda,
        microsoft_voice=microsoft_voice,
        coqui_model=coqui_model,
        coqui_speaker_wav=Path(_coerce_file_path(coqui_speaker_wav)).expanduser().resolve()
        if _coerce_file_path(coqui_speaker_wav)
        else None,
        coqui_language=coqui_language.strip() or None,
    )

    if task_norm == "Transcribe only":
        cfg.tts_backend = "none"
        cfg.make_mp3 = False

    progress(0.05, desc="Running pipeline…")

    # Prefer local file if one is uploaded, even if dropdown payload is malformed.
    if source_type_norm == "Local file" or (lf and not (youtube_url or "").strip()):
        out_dir_path = run_local_file(Path(lf), cfg) if lf else None
        if not out_dir_path:
            raise gr.Error("Please upload a local audio/video/text file.")
        return _collect_artifacts([out_dir_path])

    if source_type_norm == "YouTube":
        if not youtube_url or not youtube_url.strip():
            raise gr.Error("Please provide a YouTube video or playlist URL.")
        out_dirs = run_many(youtube_url.strip(), cfg)
        return _collect_artifacts(out_dirs)

    # Local file
    if not lf:
        raise gr.Error("Please upload a local audio/video/text file.")
    out_dir_path = run_local_file(Path(lf), cfg)
    return _collect_artifacts([out_dir_path])


def build_app(default_out: str = "out") -> gr.Blocks:
    ollama_models = list_ollama_models()
    default_polish_model = ollama_models[0] if ollama_models else ""

    with gr.Blocks(title="y2tts — YouTube/Local → ASR → TTS") as demo:
        gr.Markdown(
            "## y2tts\n"
            "Local YouTube/local-file pipeline for transcription and speech synthesis.\n\n"
            "1. Choose source and task.\n"
            "2. Provide either a YouTube URL or a local file.\n"
            "3. Adjust ASR/TTS settings only if needed.\n"
            "4. Run and download artifacts."
        )

        with gr.Group():
            gr.Markdown("### 1) Source + Task")
            with gr.Row():
                source_type = gr.Dropdown(
                    choices=["YouTube", "Local file"],
                    value="YouTube",
                    label="Source",
                    info="Pick where input comes from.",
                    scale=1,
                )
                task = gr.Dropdown(
                    choices=["Run (ASR + TTS)", "Transcribe only", "Download only (YouTube)"],
                    value="Run (ASR + TTS)",
                    label="Task",
                    info="Run full pipeline, ASR-only, or download-only.",
                    scale=1,
                )
            out_dir = gr.Textbox(
                label="Output directory",
                value=default_out,
                info="One subfolder per video/file will be created here.",
            )

        with gr.Group():
            gr.Markdown("### 2) Input")
            with gr.Row():
                youtube_url = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://youtu.be/… or https://youtube.com/playlist?list=…",
                    info="Used when Source is YouTube.",
                    scale=3,
                )
                local_file = gr.File(
                    label="Local file upload",
                    file_types=[
                        ".mp4",
                        ".mkv",
                        ".webm",
                        ".mov",
                        ".mp3",
                        ".wav",
                        ".m4a",
                        ".flac",
                        ".ogg",
                        ".aac",
                        ".txt",
                        ".vtt",
                    ],
                    type="filepath",
                    scale=2,
                )
            gr.Markdown(
                "Provide at least one input. For local `.txt`/`.vtt`, the app skips ASR and runs direct TTS."
            )

        with gr.Accordion("3) ASR Settings (faster-whisper)", open=False):
            with gr.Row():
                model = gr.Textbox(
                    label="Model",
                    value="distil-large-v3",
                    info="Examples: small, medium, large-v3, distil-large-v3.",
                )
                device = gr.Dropdown(
                    label="Device",
                    choices=["auto", "cpu", "cuda"],
                    value="auto",
                    info="Use auto unless you need to force CPU/GPU.",
                )
            with gr.Row():
                compute_type = gr.Textbox(
                    label="Compute type (optional)",
                    value="",
                    placeholder="e.g. int8, float16, int8_float16",
                )
                lang = gr.Textbox(
                    label="Language (optional)",
                    value="",
                    placeholder="e.g. en, zh, fr (leave empty for auto)",
                )
            with gr.Row():
                vad = gr.Checkbox(label="VAD filter", value=True, info="Helps remove long silence/noise.")
                word_ts = gr.Checkbox(label="Word timestamps (slower)", value=False, info="Needed only for word-level timing.")

        with gr.Accordion("4) AI Polish (optional)", open=False):
            gr.Markdown(
                "Polishing creates a separate `transcript_tts.txt`; the original and basic-clean "
                "transcripts remain unchanged. Ollama stays local. Other endpoints may receive "
                "private transcript text."
            )
            with gr.Row():
                polish_backend = gr.Dropdown(
                    label="Polish backend",
                    choices=["none", "ollama", "openai-compatible"],
                    value="none",
                    info="Use Ollama for offline polishing or a compatible API endpoint.",
                )
                polish_model = gr.Dropdown(
                    label="Polish model",
                    choices=ollama_models,
                    value=default_polish_model,
                    allow_custom_value=True,
                    info="Installed Ollama models are detected automatically; custom API model names are also allowed.",
                )
            with gr.Row():
                polish_base_url = gr.Textbox(
                    label="API base URL",
                    value="",
                    placeholder="Ollama default: http://127.0.0.1:11434",
                    info="Leave empty for Ollama's local default.",
                )
                polish_api_key_env = gr.Textbox(
                    label="API key environment variable",
                    value="OPENAI_API_KEY",
                    info="The UI reads the key from this environment variable; never paste a key here.",
                )
            with gr.Row():
                polish_glossary = gr.File(
                    label="Terminology glossary (optional)",
                    file_types=[".txt", ".md"],
                    type="filepath",
                )
                polish_chunk_chars = gr.Number(
                    label="Characters per polish request",
                    value=8000,
                    precision=0,
                    info="Smaller chunks use less model context but may reduce continuity.",
                )
                polish_timeout = gr.Number(
                    label="Per-request timeout (seconds)",
                    value=600,
                    precision=0,
                )
                ollama_num_gpu = gr.Number(
                    label="Ollama GPU layers",
                    value=0,
                    precision=0,
                    info="Use 0 for CPU-only stability; use -1 for Ollama automatic selection.",
                )
            allow_online_polish = gr.Checkbox(
                label="Allow online AI polishing",
                value=False,
                info="Required for non-local endpoints because transcript chunks will leave this computer.",
            )

        with gr.Accordion("5) TTS Settings", open=True):
            with gr.Row():
                tts = gr.Dropdown(
                    label="Backend",
                    choices=["piper", "microsoft", "coqui", "none"],
                    value="piper",
                    info="Set to 'none' for transcription-only output.",
                )
                tts_speed = gr.Slider(
                    label="Speech rate",
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.05,
                    info="1.0=normal, <1.0 slower, >1.0 faster.",
                )
                preserve_paragraph_breaks = gr.Checkbox(
                    label="Preserve paragraph breaks",
                    value=False,
                    info="Keeps paragraph pauses in cleaned transcript for TTS pacing.",
                )
                mp3 = gr.Checkbox(label="Also output mp3", value=False, info="Creates `tts.mp3` in addition to wav.")

            with gr.Accordion("Piper", open=False):
                piper_voice = gr.Textbox(label="Voice", value="en_US-lessac-medium", info="Voice name to synthesize with.")
                piper_data_dir = gr.Textbox(
                    label="Voice data dir (optional)",
                    value="",
                    placeholder="e.g. ./voices",
                    info="Directory containing local Piper voice models.",
                )
                piper_cuda = gr.Checkbox(label="Use Piper CUDA (requires onnxruntime-gpu)", value=False)

            with gr.Accordion("Microsoft Edge Neural (online)", open=False):
                microsoft_voice = gr.Textbox(
                    label="Voice",
                    value="en-US-MichelleNeural",
                    info="Online Microsoft neural voice, matching OpenClaw's Microsoft provider.",
                )

            with gr.Accordion("Coqui", open=False):
                coqui_model = gr.Textbox(
                    label="Model name",
                    value="tts_models/en/jenny/jenny",
                    info="Coqui model identifier.",
                )
                coqui_speaker_wav = gr.File(
                    label="Speaker WAV (optional, for cloning models)",
                    file_types=[".wav"],
                    type="filepath",
                )
                coqui_language = gr.Textbox(label="Language (optional, needed for some multilingual models)", value="")

        with gr.Accordion("6) Download-Only Options (YouTube)", open=False):
            # allow_custom_value avoids Gradio raising if it receives an unexpected payload
            # (some versions can send non-string values under certain UI states).
            dl_kind = gr.Dropdown(
                label="Download kind",
                choices=["video", "audio"],
                value="video",
                info="Used only when Task is Download only (YouTube).",
                allow_custom_value=True,
            )
            dl_audio_format = gr.Dropdown(
                label="Audio format (for kind=audio)",
                choices=["wav", "mp3"],
                value="wav",
                info="Used only when Download kind is audio.",
                allow_custom_value=True,
            )
            gr.Markdown("- **video**: best video+audio merged (mp4 by default)\n- **audio**: best audio extracted to `wav` or `mp3`")

        run_btn = gr.Button("Run Pipeline", variant="primary")
        status = gr.Textbox(label="Result folders", lines=4)
        preview = gr.Textbox(
            label="Transcript preview",
            lines=12,
            info="Shows the AI-polished transcript when available, otherwise the basic-clean text.",
        )
        downloads = gr.Files(label="Artifacts", file_count="multiple")

        run_btn.click(
            fn=_run,
            inputs=[
                source_type,
                youtube_url,
                local_file,
                out_dir,
                task,
                model,
                device,
                compute_type,
                lang,
                vad,
                word_ts,
                polish_backend,
                polish_model,
                polish_base_url,
                polish_api_key_env,
                polish_glossary,
                polish_chunk_chars,
                polish_timeout,
                ollama_num_gpu,
                allow_online_polish,
                tts,
                tts_speed,
                preserve_paragraph_breaks,
                mp3,
                piper_voice,
                piper_data_dir,
                piper_cuda,
                microsoft_voice,
                coqui_model,
                coqui_speaker_wav,
                coqui_language,
                dl_kind,
                dl_audio_format,
            ],
            outputs=[status, preview, downloads],
        )

        gr.Markdown(
            "### Notes\n"
            "- **Download only (YouTube)** skips ASR/TTS and just downloads media.\n"
            "- **Local file** mode accepts audio/video plus `.txt`/`.vtt` for direct TTS.\n"
            "- **Ollama polish** stays on localhost; online-compatible endpoints require explicit consent.\n"
            "- Requires `ffmpeg` on PATH."
        )

    return demo


def main(host: str = "127.0.0.1", port: int = 7860, share: bool = False, out: str = "out") -> None:
    demo = build_app(default_out=out)
    # Gradio queue API changed across versions.
    try:
        demo.queue(default_concurrency_limit=1, max_size=20)  # Gradio 5+
    except TypeError:
        try:
            demo.queue(concurrency_count=1, max_size=20)      # some Gradio 4.x builds
        except TypeError:
            demo.queue(max_size=20)
    demo.launch(server_name=host, server_port=port, share=share)
