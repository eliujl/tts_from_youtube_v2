from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .pipeline import RunConfig, download_only, run_local_file, run_many

app = typer.Typer(add_completion=False, help="Download YouTube audio -> transcribe -> resynthesize (local ASR/TTS).")
console = Console()


def _print_done(paths: list[Path]) -> None:
    tbl = Table(title="Completed")
    tbl.add_column("Output folder", overflow="fold")
    for p in paths:
        tbl.add_row(str(p))
    console.print(tbl)


@app.command()
def transcribe(
    url: str = typer.Argument(..., help="YouTube video or playlist URL"),
    out: Path = typer.Option(Path("out"), "--out", "-o", help="Output directory"),
    model: str = typer.Option("distil-large-v3", "--model", help="faster-whisper model name"),
    device: str = typer.Option("auto", "--device", help="auto|cpu|cuda"),
    lang: str | None = typer.Option(None, "--lang", help="Language code, e.g. en. Default=auto."),
    vad: bool = typer.Option(True, "--vad/--no-vad", help="Enable Silero VAD filter"),
    word_ts: bool = typer.Option(False, "--word-ts", help="Include word-level timestamps (slower)"),
):
    cfg = RunConfig(
        out_dir=out,
        asr_model=model,
        asr_device=device,
        asr_language=lang,
        asr_vad=vad,
        asr_word_timestamps=word_ts,
        tts_backend="none",
        make_mp3=False,
    )
    paths = run_many(url, cfg)
    _print_done(paths)


@app.command()
def run(
    url: str = typer.Argument(..., help="YouTube video or playlist URL"),
    out: Path = typer.Option(Path("out"), "--out", "-o", help="Output directory"),
    # ASR
    asr: str = typer.Option("faster-whisper", "--asr", help="ASR backend (currently: faster-whisper)"),
    model: str = typer.Option("distil-large-v3", "--model", help="faster-whisper model name"),
    device: str = typer.Option("auto", "--device", help="auto|cpu|cuda"),
    compute_type: str | None = typer.Option(None, "--compute-type", help="e.g. float16, int8, int8_float16"),
    lang: str | None = typer.Option(None, "--lang", help="Language code, e.g. en. Default=auto."),
    vad: bool = typer.Option(True, "--vad/--no-vad", help="Enable Silero VAD filter"),
    word_ts: bool = typer.Option(False, "--word-ts", help="Include word-level timestamps (slower)"),
    # TTS
    tts: str = typer.Option("piper", "--tts", help="piper|microsoft|coqui|none"),
    tts_speed: float = typer.Option(1.0, "--tts-speed", help="Speech rate multiplier: 1.0=normal, <1 slower, >1 faster"),
    preserve_paragraph_breaks: bool = typer.Option(
        False,
        "--preserve-paragraph-breaks/--no-preserve-paragraph-breaks",
        help="Keep paragraph breaks in cleaned transcript for more natural TTS pauses.",
    ),
    polish: str = typer.Option(
        "none", "--polish", help="AI polish backend: none|ollama|openai-compatible"
    ),
    polish_model: str = typer.Option("", "--polish-model", help="Model used for AI polishing"),
    polish_base_url: str = typer.Option(
        "", "--polish-base-url", help="API base URL; Ollama defaults to localhost:11434"
    ),
    polish_api_key_env: str = typer.Option(
        "OPENAI_API_KEY", "--polish-api-key-env", help="Environment variable containing API key"
    ),
    polish_glossary: Path | None = typer.Option(
        None, "--polish-glossary", help="Optional UTF-8 terminology glossary"
    ),
    polish_chunk_chars: int = typer.Option(
        8000, "--polish-chunk-chars", help="Maximum source characters per polish request"
    ),
    polish_timeout: int = typer.Option(
        600, "--polish-timeout", help="Timeout in seconds for each polish request"
    ),
    ollama_num_gpu: int | None = typer.Option(
        None,
        "--ollama-num-gpu",
        help="Ollama GPU layers; use 0 for CPU-only stability on older GPUs",
    ),
    allow_online_polish: bool = typer.Option(
        False,
        "--allow-online-polish",
        help="Allow transcript text to be sent to a non-local polish endpoint",
    ),
    mp3: bool = typer.Option(False, "--mp3", help="Also create mp3 output (requires ffmpeg + libmp3lame)"),
    piper_voice: str = typer.Option("en_US-lessac-medium", "--piper-voice", help="Piper voice name"),
    piper_data_dir: Path | None = typer.Option(None, "--piper-data-dir", help="Piper voice directory"),
    piper_cuda: bool = typer.Option(False, "--piper-cuda", help="Use Piper CUDA (requires onnxruntime-gpu)"),
    microsoft_voice: str = typer.Option(
        "en-US-MichelleNeural", "--microsoft-voice", help="Microsoft Edge neural voice"
    ),
    coqui_model: str = typer.Option("tts_models/en/jenny/jenny", "--coqui-model", help="Coqui model name"),
    coqui_speaker_wav: Path | None = typer.Option(None, "--coqui-speaker-wav", help="Speaker wav for cloning"),
    coqui_language: str | None = typer.Option(None, "--coqui-language", help="Language for multilingual models"),
):
    cfg = RunConfig(
        out_dir=out,
        asr_backend="faster-whisper",
        asr_model=model,
        asr_device=device,
        asr_compute_type=compute_type,
        asr_language=lang,
        asr_vad=vad,
        asr_word_timestamps=word_ts,
        tts_backend=tts,  # type: ignore[arg-type]
        tts_speed=tts_speed,
        preserve_paragraph_breaks=preserve_paragraph_breaks,
        polish_backend=polish,  # type: ignore[arg-type]
        polish_model=polish_model,
        polish_base_url=polish_base_url,
        polish_api_key_env=polish_api_key_env,
        polish_glossary_path=polish_glossary,
        polish_chunk_chars=polish_chunk_chars,
        polish_timeout_seconds=polish_timeout,
        polish_allow_remote=allow_online_polish,
        polish_ollama_num_gpu=ollama_num_gpu,
        make_mp3=mp3,
        piper_voice=piper_voice,
        piper_data_dir=piper_data_dir,
        piper_use_cuda=piper_cuda,
        microsoft_voice=microsoft_voice,
        coqui_model=coqui_model,
        coqui_speaker_wav=coqui_speaker_wav,
        coqui_language=coqui_language,
    )
    paths = run_many(url, cfg)
    _print_done(paths)

@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host (default: localhost)"),
    port: int = typer.Option(7860, "--port", help="Bind port"),
    share: bool = typer.Option(False, "--share", help="Create a public Gradio share link"),
    out: Path = typer.Option(Path("out"), "--out", "-o", help="Default output directory shown in UI"),
):
    """Launch a local web UI (Gradio). Requires: pip install -e ".[ui,asr,tts_piper]" """
    from .ui import main as ui_main

    ui_main(host=host, port=port, share=share, out=str(out))

@app.command()
def download(
    url: str = typer.Argument(..., help="YouTube video or playlist URL"),
    out: Path = typer.Option(Path("out"), "--out", "-o", help="Output directory"),
    kind: str = typer.Option("video", "--kind", help="video|audio"),
    audio_format: str = typer.Option("wav", "--audio-format", help="For --kind audio: wav|mp3"),
):
    """Download YouTube video/playlist only (no ASR/TTS)."""
    cfg = RunConfig(out_dir=out, tts_backend="none", make_mp3=False)
    paths = download_only(url, cfg, kind=kind, audio_format=audio_format)
    _print_done(paths)


@app.command()
def local(
    path: Path = typer.Argument(..., help="Local audio/video/text file path (e.g. mp4, wav, mp3, txt, vtt)"),
    out: Path = typer.Option(Path("out"), "--out", "-o", help="Output directory"),
    model: str = typer.Option("distil-large-v3", "--model", help="faster-whisper model name"),
    device: str = typer.Option("auto", "--device", help="auto|cpu|cuda"),
    compute_type: str | None = typer.Option(None, "--compute-type", help="e.g. float16, int8, int8_float16"),
    lang: str | None = typer.Option(None, "--lang", help="Language code, e.g. en. Default=auto."),
    vad: bool = typer.Option(True, "--vad/--no-vad", help="Enable Silero VAD filter"),
    word_ts: bool = typer.Option(False, "--word-ts", help="Include word-level timestamps (slower)"),
    tts: str = typer.Option("piper", "--tts", help="piper|microsoft|coqui|none"),
    tts_speed: float = typer.Option(1.0, "--tts-speed", help="Speech rate multiplier: 1.0=normal, <1 slower, >1 faster"),
    preserve_paragraph_breaks: bool = typer.Option(
        False,
        "--preserve-paragraph-breaks/--no-preserve-paragraph-breaks",
        help="Keep paragraph breaks in cleaned transcript for more natural TTS pauses.",
    ),
    polish: str = typer.Option(
        "none", "--polish", help="AI polish backend: none|ollama|openai-compatible"
    ),
    polish_model: str = typer.Option("", "--polish-model", help="Model used for AI polishing"),
    polish_base_url: str = typer.Option(
        "", "--polish-base-url", help="API base URL; Ollama defaults to localhost:11434"
    ),
    polish_api_key_env: str = typer.Option(
        "OPENAI_API_KEY", "--polish-api-key-env", help="Environment variable containing API key"
    ),
    polish_glossary: Path | None = typer.Option(
        None, "--polish-glossary", help="Optional UTF-8 terminology glossary"
    ),
    polish_chunk_chars: int = typer.Option(
        8000, "--polish-chunk-chars", help="Maximum source characters per polish request"
    ),
    polish_timeout: int = typer.Option(
        600, "--polish-timeout", help="Timeout in seconds for each polish request"
    ),
    ollama_num_gpu: int | None = typer.Option(
        None,
        "--ollama-num-gpu",
        help="Ollama GPU layers; use 0 for CPU-only stability on older GPUs",
    ),
    allow_online_polish: bool = typer.Option(
        False,
        "--allow-online-polish",
        help="Allow transcript text to be sent to a non-local polish endpoint",
    ),
    mp3: bool = typer.Option(False, "--mp3", help="Also create mp3 output (requires ffmpeg + libmp3lame)"),
    piper_voice: str = typer.Option("en_US-lessac-medium", "--piper-voice", help="Piper voice name"),
    piper_data_dir: Path | None = typer.Option(None, "--piper-data-dir", help="Piper voice directory"),
    piper_cuda: bool = typer.Option(False, "--piper-cuda", help="Use Piper CUDA (requires onnxruntime-gpu)"),
    microsoft_voice: str = typer.Option(
        "en-US-MichelleNeural", "--microsoft-voice", help="Microsoft Edge neural voice"
    ),
    coqui_model: str = typer.Option("tts_models/en/jenny/jenny", "--coqui-model", help="Coqui model name"),
    coqui_speaker_wav: Path | None = typer.Option(None, "--coqui-speaker-wav", help="Speaker wav for cloning"),
    coqui_language: str | None = typer.Option(None, "--coqui-language", help="Language for multilingual models"),
):
    """Run ASR/TTS on a local file (no YouTube). .txt/.vtt inputs skip ASR and run direct TTS."""
    cfg = RunConfig(
        out_dir=out,
        asr_model=model,
        asr_device=device,
        asr_compute_type=compute_type,
        asr_language=lang,
        asr_vad=vad,
        asr_word_timestamps=word_ts,
        tts_backend=tts,  # type: ignore[arg-type]
        tts_speed=tts_speed,
        preserve_paragraph_breaks=preserve_paragraph_breaks,
        polish_backend=polish,  # type: ignore[arg-type]
        polish_model=polish_model,
        polish_base_url=polish_base_url,
        polish_api_key_env=polish_api_key_env,
        polish_glossary_path=polish_glossary,
        polish_chunk_chars=polish_chunk_chars,
        polish_timeout_seconds=polish_timeout,
        polish_allow_remote=allow_online_polish,
        polish_ollama_num_gpu=ollama_num_gpu,
        make_mp3=mp3,
        piper_voice=piper_voice,
        piper_data_dir=piper_data_dir,
        piper_use_cuda=piper_cuda,
        microsoft_voice=microsoft_voice,
        coqui_model=coqui_model,
        coqui_speaker_wav=coqui_speaker_wav,
        coqui_language=coqui_language,
    )
    out_dir_path = run_local_file(path, cfg)
    _print_done([out_dir_path])
