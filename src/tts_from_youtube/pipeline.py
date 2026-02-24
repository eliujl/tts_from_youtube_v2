from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .audio import normalize_for_asr, wav_to_mp3
from .asr.faster_whisper_asr import Transcript, transcribe_faster_whisper
from .download import DownloadResult, DownloadVideoResult, VideoItem, download_best_audio, download_video, expand_url
from .text import basic_cleanup, load_text_input
from .utils import ensure_dir, sanitize_filename


ASRBackend = Literal["faster-whisper"]
TTSBackend = Literal["none", "piper", "coqui"]


@dataclass
class RunConfig:
    out_dir: Path
    asr_backend: ASRBackend = "faster-whisper"
    asr_model: str = "distil-large-v3"
    asr_device: str = "auto"
    asr_compute_type: str | None = None
    asr_language: str | None = None
    asr_vad: bool = True
    asr_word_timestamps: bool = False

    tts_backend: TTSBackend = "piper"
    piper_voice: str = "en_US-lessac-medium"
    piper_data_dir: Path | None = None
    piper_use_cuda: bool = False
    coqui_model: str = "tts_models/en/jenny/jenny"
    coqui_speaker_wav: Path | None = None
    coqui_language: str | None = None

    prefer_captions: bool = True
    download_captions: bool = True
    make_mp3: bool = False


def _write_manifest(out_dir: Path, payload: dict) -> None:
    (out_dir / "manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_transcript(out_dir: Path, t: Transcript) -> None:
    (out_dir / "transcript.txt").write_text(t.to_text(), encoding="utf-8")
    (out_dir / "transcript.srt").write_text(t.to_srt(), encoding="utf-8")
    # JSON with segments + optional word timestamps
    data = {
        "language": t.language,
        "language_probability": t.language_probability,
        "segments": [
            {"start": s.start, "end": s.end, "text": s.text, "words": s.words} for s in t.segments
        ],
    }
    (out_dir / "transcript.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _transcribe(audio_wav: Path, cfg: RunConfig) -> Transcript:
    if cfg.asr_backend == "faster-whisper":
        return transcribe_faster_whisper(
            audio_wav,
            model=cfg.asr_model,
            device=cfg.asr_device,
            compute_type=cfg.asr_compute_type,
            language=cfg.asr_language,
            vad_filter=cfg.asr_vad,
            word_timestamps=cfg.asr_word_timestamps,
        )
    raise ValueError(f"Unknown ASR backend: {cfg.asr_backend}")


def _synthesize(text: str, out_dir: Path, cfg: RunConfig) -> Path | None:
    if cfg.tts_backend == "none":
        return None

    out_wav = out_dir / "tts.wav"

    if cfg.tts_backend == "piper":
        from .tts.piper_tts import PiperConfig, synthesize_to_wav

        synthesize_to_wav(
            text,
            out_wav,
            PiperConfig(
                voice=cfg.piper_voice,
                data_dir=cfg.piper_data_dir,
                use_cuda=cfg.piper_use_cuda,
            ),
        )

    elif cfg.tts_backend == "coqui":
        from .tts.coqui_tts import CoquiConfig, synthesize_to_wav

        synthesize_to_wav(
            text,
            out_wav,
            CoquiConfig(
                model_name=cfg.coqui_model,
                speaker_wav=cfg.coqui_speaker_wav,
                language=cfg.coqui_language,
            ),
        )
    else:
        raise ValueError(f"Unknown TTS backend: {cfg.tts_backend}")

    if cfg.make_mp3:
        mp3 = out_dir / "tts.mp3"
        wav_to_mp3(out_wav, mp3)
        return mp3

    return out_wav


def run_single(url: str, cfg: RunConfig) -> Path:
    ensure_dir(cfg.out_dir)

    items = expand_url(url)
    if len(items) != 1:
        raise ValueError("run_single expects a single video URL; use run_many for playlists.")

    item = items[0]
    title = sanitize_filename(item.title or "video")

    work_dir = ensure_dir(cfg.out_dir / title)
    dl: DownloadResult = download_best_audio(item, work_dir, download_captions=cfg.download_captions)

    # Optional normalization for ASR (creates normalized.wav)
    norm_wav = work_dir / "audio_16k.wav"
    normalize_for_asr(dl.audio_path, norm_wav)

    t = _transcribe(norm_wav, cfg)
    _save_transcript(work_dir, t)

    cleaned = basic_cleanup(t.to_text())
    (work_dir / "transcript_clean.txt").write_text(cleaned + "\n", encoding="utf-8")

    tts_path = _synthesize(cleaned, work_dir, cfg)

    _write_manifest(
        work_dir,
        {
            "source_url": url,
            "title": dl.title,
            "video_id": dl.video_id,
            "audio_wav": str(dl.audio_path),
            "audio_norm_wav": str(norm_wav),
            "captions": str(dl.captions_path) if dl.captions_path else None,
            "asr": {
                "backend": cfg.asr_backend,
                "model": cfg.asr_model,
                "language": t.language,
                "language_probability": t.language_probability,
            },
            "tts": {"backend": cfg.tts_backend, "output": str(tts_path) if tts_path else None},
        },
    )

    return work_dir


def run_local_file(path: Path, cfg: RunConfig) -> Path:
    """Process a local audio/video file.

    - Extract/resample audio to 16k mono wav via ffmpeg
    - ASR (optional)
    - TTS (optional)
    """
    ensure_dir(cfg.out_dir)

    src = Path(path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(str(src))

    title = sanitize_filename(src.stem)
    work_dir = ensure_dir(cfg.out_dir / title)

    # Text-first path: no ASR needed, synthesize directly from provided transcript.
    if src.suffix.lower() in {".txt", ".vtt"}:
        raw_text = load_text_input(src)
        if not raw_text:
            raise ValueError(f"No usable text found in input file: {src}")

        (work_dir / "transcript.txt").write_text(raw_text + "\n", encoding="utf-8")
        cleaned = basic_cleanup(raw_text)
        (work_dir / "transcript_clean.txt").write_text(cleaned + "\n", encoding="utf-8")
        tts_path = _synthesize(cleaned, work_dir, cfg)

        _write_manifest(
            work_dir,
            {
                "source_file": str(src),
                "title": title,
                "input_kind": "text",
                "tts": {"backend": cfg.tts_backend, "output": str(tts_path) if tts_path else None},
            },
        )
        return work_dir

    # Normalize/extract audio for ASR
    norm_wav = work_dir / "audio_16k.wav"
    normalize_for_asr(src, norm_wav)

    t = _transcribe(norm_wav, cfg)
    _save_transcript(work_dir, t)

    cleaned = basic_cleanup(t.to_text())
    (work_dir / "transcript_clean.txt").write_text(cleaned + "\n", encoding="utf-8")

    tts_path = _synthesize(cleaned, work_dir, cfg)

    _write_manifest(
        work_dir,
        {
            "source_file": str(src),
            "title": title,
            "audio_norm_wav": str(norm_wav),
            "asr": {
                "backend": cfg.asr_backend,
                "model": cfg.asr_model,
                "language": t.language,
                "language_probability": t.language_probability,
            },
            "tts": {"backend": cfg.tts_backend, "output": str(tts_path) if tts_path else None},
        },
    )

    return work_dir


def download_only(url: str, cfg: RunConfig, *, kind: str = "video") -> list[Path]:
    """Download YouTube video/playlist without ASR/TTS.

    kind:
      - "video": best video+audio merged
      - "audio": best audio as wav (uses the existing audio downloader)
    """
    ensure_dir(cfg.out_dir)
    items = expand_url(url)

    out_paths: list[Path] = []
    for item in items:
        if kind == "video":
            # Put each item in its own folder
            title = sanitize_filename(item.title or "video")
            work_dir = ensure_dir(cfg.out_dir / title)
            v = download_video(item, work_dir)
            _write_manifest(
                work_dir,
                {
                    "source_url": item.url,
                    "title": v.title,
                    "video_id": v.video_id,
                    "download_kind": "video",
                    "video_path": str(v.video_path),
                },
            )
            out_paths.append(work_dir)
        elif kind == "audio":
            title = sanitize_filename(item.title or "audio")
            work_dir = ensure_dir(cfg.out_dir / title)
            a = download_best_audio(item, work_dir, download_captions=False)
            _write_manifest(
                work_dir,
                {
                    "source_url": item.url,
                    "title": a.title,
                    "video_id": a.video_id,
                    "download_kind": "audio",
                    "audio_wav": str(a.audio_path),
                },
            )
            out_paths.append(work_dir)
        else:
            raise ValueError("kind must be 'video' or 'audio'")

    return out_paths



def run_many(url: str, cfg: RunConfig) -> list[Path]:
    ensure_dir(cfg.out_dir)
    items = expand_url(url)
    out_paths: list[Path] = []
    for item in items:
        # Use the item URL as a single video run (playlist already expanded)
        out_paths.append(run_single(item.url, cfg))
    return out_paths
