from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class CoquiConfig:
    model_name: str = "tts_models/en/jenny/jenny"
    speaker_wav: Path | None = None
    language: str | None = None  # required for some multilingual models e.g. xtts_v2


def synthesize_to_wav(text: str, out_wav: Path, cfg: CoquiConfig) -> None:
    from TTS.api import TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    tts = TTS(cfg.model_name, progress_bar=True).to(device)

    if cfg.speaker_wav is not None:
        if cfg.language is None:
            raise ValueError("Coqui voice cloning models require --coqui-language when speaker_wav is set.")
        tts.tts_to_file(
            text=text,
            speaker_wav=str(cfg.speaker_wav),
            language=cfg.language,
            file_path=str(out_wav),
        )
    else:
        tts.tts_to_file(text=text, file_path=str(out_wav))
