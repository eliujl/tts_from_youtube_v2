from __future__ import annotations

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
