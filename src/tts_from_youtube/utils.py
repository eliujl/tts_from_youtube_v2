from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

_INVALID_FILENAME = re.compile(r'[\\/:*?"<>|]+')


def sanitize_filename(name: str, max_len: int = 180) -> str:
    name = _INVALID_FILENAME.sub("_", name).strip()
    name = re.sub(r"\s+", " ", name).strip()
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name or "untitled"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def which(cmd: str) -> str | None:
    for folder in os.environ.get("PATH", "").split(os.pathsep):
        c = Path(folder) / cmd
        if os.name == "nt":
            for ext in (".exe", ".bat", ".cmd"):
                if (c.with_suffix(ext)).exists():
                    return str(c.with_suffix(ext))
        if c.exists():
            return str(c)
    return None
