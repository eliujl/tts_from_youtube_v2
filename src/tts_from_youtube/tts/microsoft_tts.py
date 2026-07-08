from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..audio import concat_wavs, require_ffmpeg


@dataclass
class MicrosoftConfig:
    voice: str = "en-US-MichelleNeural"
    speed: float = 1.0


_SENTENCE_BREAK = re.compile(r"(?<=[。！？!?；;:：,.，])")
_DEFAULT_MAX_CHARS = 1500


def _rate_percent(speed: float) -> str:
    if speed <= 0:
        raise ValueError("speed must be > 0.")
    return f"{round((speed - 1.0) * 100):+d}%"


def _edge_command() -> tuple[list[str], dict[str, str] | None]:
    """Return an edge-tts command, including the local fallback installation."""
    try:
        import edge_tts  # noqa: F401

        return [sys.executable, "-m", "edge_tts"], None
    except ImportError:
        pass

    # Development fallback for workspaces whose existing virtualenv is read-only.
    project_root = Path(__file__).resolve().parents[3]
    local_packages = project_root / ".edge_tts_packages"
    python = shutil.which("python")
    if local_packages.is_dir() and python:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(local_packages) + (os.pathsep + existing if existing else "")
        return [python, "-m", "edge_tts"], env

    raise RuntimeError(
        'Microsoft TTS requires edge-tts. Install it with: pip install -e ".[tts_microsoft]"'
    )


def split_text_for_microsoft(text: str, max_chars: int = _DEFAULT_MAX_CHARS) -> list[str]:
    """Split long text into smaller chunks for more reliable Edge TTS synthesis."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", normalized) if part.strip()]
    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for paragraph in paragraphs:
        parts = _SENTENCE_BREAK.split(paragraph)
        if not parts:
            parts = [paragraph]
        for part in parts:
            piece = part.strip()
            if not piece:
                continue
            if len(piece) > max_chars:
                flush()
                for start in range(0, len(piece), max_chars):
                    sub = piece[start : start + max_chars].strip()
                    if sub:
                        chunks.append(sub)
                continue
            candidate = f"{current}\n\n{piece}".strip() if current else piece
            if len(candidate) <= max_chars:
                current = candidate
            else:
                flush()
                current = piece
    flush()
    return chunks


def synthesize_to_wav(text: str, out_wav: Path, cfg: MicrosoftConfig) -> None:
    """Synthesize with Microsoft Edge Neural TTS and convert its MP3 to WAV."""
    require_ffmpeg()
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    command, env = _edge_command()
    chunks = split_text_for_microsoft(text)
    if not chunks:
        raise ValueError("No text was available for Microsoft TTS synthesis.")

    with tempfile.TemporaryDirectory(prefix="microsoft-tts-", dir=out_wav.parent) as temp_dir:
        temp_path = Path(temp_dir)
        wav_parts: list[Path] = []
        for index, chunk in enumerate(chunks, start=1):
            input_txt = temp_path / f"input_{index:04d}.txt"
            output_mp3 = temp_path / f"output_{index:04d}.mp3"
            output_wav = temp_path / f"output_{index:04d}.wav"
            input_txt.write_text(chunk, encoding="utf-8")

            subprocess.run(
                [
                    *command,
                    "--file",
                    str(input_txt),
                    "--voice",
                    cfg.voice,
                    f"--rate={_rate_percent(cfg.speed)}",
                    "--write-media",
                    str(output_mp3),
                ],
                check=True,
                env=env,
            )
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(output_mp3), "-c:a", "pcm_s16le", str(output_wav)],
                check=True,
            )
            wav_parts.append(output_wav)

        concat_wavs(wav_parts, out_wav)
