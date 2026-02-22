from __future__ import annotations

import subprocess
from pathlib import Path

from .utils import which


class FFmpegMissing(RuntimeError):
    pass


def require_ffmpeg() -> None:
    if which("ffmpeg") is None:
        raise FFmpegMissing(
            "ffmpeg not found on PATH. Install ffmpeg and ensure it's accessible from the shell."
        )


def wav_to_mp3(wav_path: Path, mp3_path: Path, bitrate: str = "192k") -> None:
    require_ffmpeg()
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(wav_path),
        "-vn",
        "-c:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        str(mp3_path),
    ]
    subprocess.run(cmd, check=True)


def normalize_for_asr(in_path: Path, out_wav_16k_mono: Path) -> None:
    """Optional normalization step: resample to 16kHz mono PCM for more predictable ASR."""
    require_ffmpeg()
    out_wav_16k_mono.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(in_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(out_wav_16k_mono),
    ]
    subprocess.run(cmd, check=True)
