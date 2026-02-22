from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import datetime
from typing import Any

import srt


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[dict[str, Any]] | None = None


@dataclass
class Transcript:
    language: str | None
    language_probability: float | None
    segments: list[Segment]

    def to_text(self) -> str:
        return "\n".join(s.text.strip() for s in self.segments).strip() + "\n"

    def to_srt(self) -> str:
        subs = []
        for i, seg in enumerate(self.segments, start=1):
            subs.append(
                srt.Subtitle(
                    index=i,
                    start=datetime.timedelta(seconds=float(seg.start)),
                    end=datetime.timedelta(seconds=float(seg.end)),
                    content=seg.text.strip(),
                )
            )
        return srt.compose(subs)


def transcribe_faster_whisper(
    audio_path: Path,
    *,
    model: str = "distil-large-v3",
    device: str = "auto",
    compute_type: str | None = None,
    language: str | None = None,
    vad_filter: bool = True,
    word_timestamps: bool = False,
    beam_size: int = 5,
    condition_on_previous_text: bool = False,
) -> Transcript:
    """Transcribe with faster-whisper.

    Notes:
    - device: "auto"|"cpu"|"cuda"
    - compute_type: e.g. "int8", "float16", "int8_float16"
    """
    from faster_whisper import WhisperModel

    if device == "auto":
        # Faster-Whisper handles "cuda" only if the underlying CTranslate2 build supports it.
        # We keep it simple: prefer CUDA if available via torch.
        try:
            import torch  # type: ignore

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    fw = WhisperModel(model, device=device, compute_type=compute_type)
    segments_iter, info = fw.transcribe(
        str(audio_path),
        beam_size=beam_size,
        language=language,
        vad_filter=vad_filter,
        word_timestamps=word_timestamps,
        condition_on_previous_text=condition_on_previous_text,
    )

    segments: list[Segment] = []
    for seg in segments_iter:
        words = None
        if getattr(seg, "words", None):
            words = [{"start": w.start, "end": w.end, "word": w.word, "probability": w.probability} for w in seg.words]
        segments.append(Segment(start=seg.start, end=seg.end, text=seg.text, words=words))

    return Transcript(
        language=getattr(info, "language", None),
        language_probability=getattr(info, "language_probability", None),
        segments=segments,
    )
