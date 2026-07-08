from __future__ import annotations

import subprocess
import tempfile
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


def _build_atempo_chain(speed: float) -> str:
    """Build ffmpeg atempo chain with per-stage factor constrained to [0.5, 2.0]."""
    if speed <= 0:
        raise ValueError("speed must be > 0.")

    parts: list[str] = []
    remaining = speed
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    while remaining > 2.0:
        parts.append("atempo=2.0")
        remaining /= 2.0
    parts.append(f"atempo={remaining:.6f}")
    return ",".join(parts)


def retime_wav(in_wav: Path, out_wav: Path, speed: float) -> None:
    """Adjust speaking speed using ffmpeg atempo, preserving pitch."""
    if speed <= 0:
        raise ValueError("speed must be > 0.")
    if speed == 1.0:
        return

    require_ffmpeg()
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(in_wav),
        "-filter:a",
        _build_atempo_chain(speed),
        str(out_wav),
    ]
    subprocess.run(cmd, check=True)


def concat_wavs(inputs: list[Path], out_wav: Path) -> None:
    """Concatenate WAV files with ffmpeg."""
    if not inputs:
        raise ValueError("concat_wavs requires at least one input file.")
    if len(inputs) == 1:
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        if inputs[0] != out_wav:
            out_wav.write_bytes(inputs[0].read_bytes())
        return

    require_ffmpeg()
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ffmpeg-concat-", dir=out_wav.parent) as temp_dir:
        list_path = Path(temp_dir) / "inputs.txt"
        list_path.write_text(
            "".join(f"file '{p.resolve().as_posix()}'\n" for p in inputs),
            encoding="utf-8",
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c:a",
            "pcm_s16le",
            str(out_wav),
        ]
        subprocess.run(cmd, check=True)
