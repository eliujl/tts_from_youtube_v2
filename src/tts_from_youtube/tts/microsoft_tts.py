from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..audio import require_ffmpeg


@dataclass
class MicrosoftConfig:
    voice: str = "en-US-MichelleNeural"
    speed: float = 1.0


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


def synthesize_to_wav(text: str, out_wav: Path, cfg: MicrosoftConfig) -> None:
    """Synthesize with Microsoft Edge Neural TTS and convert its MP3 to WAV."""
    require_ffmpeg()
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    command, env = _edge_command()

    with tempfile.TemporaryDirectory(prefix="microsoft-tts-", dir=out_wav.parent) as temp_dir:
        temp_path = Path(temp_dir)
        input_txt = temp_path / "input.txt"
        output_mp3 = temp_path / "output.mp3"
        input_txt.write_text(text, encoding="utf-8")

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
            ["ffmpeg", "-y", "-i", str(output_mp3), "-c:a", "pcm_s16le", str(out_wav)],
            check=True,
        )
