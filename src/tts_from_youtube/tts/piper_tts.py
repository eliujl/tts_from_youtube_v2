from __future__ import annotations

import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

from ..utils import ensure_dir


@dataclass
class PiperConfig:
    voice: str = "en_US-lessac-medium"
    data_dir: Path | None = None
    use_cuda: bool = False
    length_scale: float | None = None  # >1.0 slower, <1.0 faster
    noise_scale: float | None = None
    noise_w_scale: float | None = None
    volume: float | None = None


def ensure_voice_downloaded(cfg: PiperConfig) -> Path:
    """Download a Piper voice if it's not present. Returns the ONNX model path."""
    from piper import PiperVoice  # noqa: F401

    data_dir = cfg.data_dir or Path.cwd() / "voices"
    ensure_dir(data_dir)

    # Voice package downloader stores in data_dir/<voice>/
    voice_dir = data_dir / cfg.voice
    onnx = voice_dir / f"{cfg.voice}.onnx"
    if onnx.exists():
        return onnx

    cmd = [sys.executable, "-m", "piper.download_voices", cfg.voice, "--data-dir", str(data_dir)]
    subprocess.run(cmd, check=True)
    if not onnx.exists():
        # Some distributions download to cwd; fall back to search.
        candidates = list(data_dir.rglob("*.onnx"))
        if not candidates:
            raise RuntimeError(f"Failed to download Piper voice: {cfg.voice}")
        return candidates[0]
    return onnx


def synthesize_to_wav(text: str, out_wav: Path, cfg: PiperConfig) -> None:
    from piper import PiperVoice
    from piper.config import SynthesisConfig  # type: ignore

    model_path = ensure_voice_downloaded(cfg)
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    syn_cfg = None
    if any(v is not None for v in [cfg.length_scale, cfg.noise_scale, cfg.noise_w_scale, cfg.volume]):
        syn_cfg = SynthesisConfig(
            length_scale=cfg.length_scale if cfg.length_scale is not None else 1.0,
            noise_scale=cfg.noise_scale if cfg.noise_scale is not None else 0.667,
            noise_w_scale=cfg.noise_w_scale if cfg.noise_w_scale is not None else 0.8,
            volume=cfg.volume if cfg.volume is not None else 1.0,
            normalize_audio=True,
        )

    voice = PiperVoice.load(str(model_path), use_cuda=cfg.use_cuda)
    with wave.open(str(out_wav), "wb") as f:
        voice.synthesize_wav(text, f, syn_config=syn_cfg)
