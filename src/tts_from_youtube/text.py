from __future__ import annotations

from pathlib import Path

import regex as re

_FILLS = re.compile(
    r"\b(um+|uh+|er+|ah+|like|you\s+know|i\s+mean)\b",
    flags=re.IGNORECASE,
)


def basic_cleanup(text: str, *, remove_fillers: bool = True) -> str:
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    if remove_fillers:
        text = _FILLS.sub("", text)
        text = re.sub(r"\s+", " ", text).strip()
    return text


_VTT_TIMESTAMP = re.compile(
    r"^\s*\d{2}:\d{2}(?::\d{2})?\.\d{3}\s+-->\s+\d{2}:\d{2}(?::\d{2})?\.\d{3}"
)
_VTT_CUE_INDEX = re.compile(r"^\s*\d+\s*$")


def load_text_input(path: Path) -> str:
    """Load plain text from a .txt or .vtt file."""
    src = Path(path).expanduser().resolve()
    suffix = src.suffix.lower()

    if suffix == ".txt":
        return src.read_text(encoding="utf-8").strip()

    if suffix == ".vtt":
        lines = src.read_text(encoding="utf-8").splitlines()
        text_lines: list[str] = []
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            if line.upper() == "WEBVTT":
                continue
            if line.startswith("NOTE"):
                continue
            if _VTT_TIMESTAMP.match(line):
                continue
            if _VTT_CUE_INDEX.match(line):
                continue
            text_lines.append(line)
        return " ".join(text_lines).strip()

    raise ValueError(f"Unsupported text input type: {suffix}")
