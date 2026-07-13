from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

PolishBackend = Literal["none", "ollama", "openai-compatible"]


class PolishError(RuntimeError):
    pass


class PolishFidelityError(PolishError):
    """The model rewrite changed or dropped source information."""


@dataclass
class PolishConfig:
    backend: PolishBackend = "none"
    model: str = ""
    base_url: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    glossary_path: Path | None = None
    instructions_path: Path | None = None
    chunk_chars: int = 8000
    timeout_seconds: int = 600
    allow_remote: bool = False
    checkpoint_dir: Path | None = None
    ollama_num_gpu: int | None = None


_SYSTEM_PROMPT = """You are a conservative transcript editor preparing text for speech synthesis.
Preserve every factual claim, teaching, example, question, answer, and their original order.
Do not summarize, add facts, modernize the ideas, or silently resolve genuine ambiguity.
Keep numeric expressions and quantities exactly as written in the source.
Repair likely transcription errors, punctuation, sentence boundaries, grammar, false starts,
and accidental repetition. Use short, natural sentences and paragraph breaks for audible pacing.
Keep meaningful speaker context. Omit only chatter that is wholly unintelligible and nonessential.
Remove timestamps such as 2021-01-11 22:52:52 when they are page or metadata noise rather than
spoken content. Remove formatting symbols such as **, ##, -----, and similar page-only markup.
Omit tables of contents, page-navigation entries, URLs, HTTP links, link/share metadata, machine
timestamps or IDs, and other material that is useful on a page but unsuitable for spoken audio.
For Chinese text, remove inappropriate spaces between Chinese characters and punctuation.
Expand symbols or abbreviations only when needed for natural speech.
Return only the polished transcript text, with no commentary, heading, or Markdown fence."""

_NUMBER_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "million",
    "billion",
}
_DIGIT_WORDS = {
    str(index): word
    for index, word in enumerate(
        [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
        ]
    )
}
_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "because",
    "been",
    "before",
    "being",
    "could",
    "from",
    "have",
    "into",
    "just",
    "more",
    "only",
    "other",
    "should",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "very",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "your",
}


def _is_loopback_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def list_ollama_models(base_url: str = "http://127.0.0.1:11434") -> list[str]:
    """Return installed text-generation models, ordered for transcript polishing."""
    base_url = base_url.strip() or "http://127.0.0.1:11434"
    if base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/")[:-3]
    if not _is_loopback_url(base_url):
        return []

    request = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return []

    candidates: list[tuple[str, int]] = []
    for item in body.get("models", []):
        name = str(item.get("name", "")).strip()
        size = int(item.get("size", 0) or 0)
        if name and "embed" not in name.lower():
            candidates.append((name, size))

    # Prefer the locally verified 4B Qwen model when present. Otherwise favor a
    # roughly 4 GB model: large enough to edit faithfully, but practical on CPU.
    candidates.sort(
        key=lambda item: (
            item[0].lower() != "qwen3.5:4b",
            abs(item[1] - 4_000_000_000),
            item[0].lower(),
        )
    )
    return [name for name, _ in candidates]


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def split_for_polish(text: str, max_chars: int) -> list[str]:
    """Split at paragraph/sentence boundaries while keeping every source character."""
    if max_chars < 500:
        raise ValueError("polish chunk size must be at least 500 characters")

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            units.append(paragraph)
            continue

        sentence_group: list[str] = []
        sentence_length = 0
        for sentence in _sentences(paragraph):
            extra = len(sentence) + (1 if sentence_group else 0)
            if sentence_group and sentence_length + extra > max_chars:
                units.append(" ".join(sentence_group))
                sentence_group = []
                sentence_length = 0

            if len(sentence) > max_chars:
                if sentence_group:
                    units.append(" ".join(sentence_group))
                    sentence_group = []
                    sentence_length = 0
                units.extend(
                    sentence[start : start + max_chars]
                    for start in range(0, len(sentence), max_chars)
                )
            else:
                sentence_group.append(sentence)
                sentence_length += extra

        if sentence_group:
            units.append(" ".join(sentence_group))

    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for unit in units:
        extra = len(unit) + (2 if current else 0)
        if current and current_length + extra > max_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_length = 0
        current.append(unit)
        current_length += extra
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _strip_model_wrapping(text: str) -> str:
    result = text.strip()
    if result.startswith("```") and result.endswith("```"):
        result = re.sub(r"^```(?:text|markdown)?\s*", "", result, flags=re.IGNORECASE)
        result = re.sub(r"\s*```$", "", result)
    return result.strip()


def _number_tokens(text: str) -> Counter[str]:
    tokens = re.findall(r"\b(?:\d+|[a-z]+)\b", text.lower())
    normalized = [_DIGIT_WORDS.get(token, token) for token in tokens]
    return Counter(token for token in normalized if token in _NUMBER_WORDS or token.isdigit())


def _stem_token(token: str) -> str:
    for suffix in ("ingly", "edly", "ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def _content_tokens(text: str) -> set[str]:
    return {
        _stem_token(token)
        for token in re.findall(r"\b[a-z][a-z'-]+\b", text.lower())
        if len(token) >= 4 and token not in _STOPWORDS and token not in _NUMBER_WORDS
    }


def _validate_polished_chunk(source: str, polished: str, index: int) -> None:
    length_ratio = len(polished) / max(1, len(source))
    if not 0.35 <= length_ratio <= 2.0:
        raise PolishFidelityError(
            f"Polished chunk {index} changed length suspiciously "
            f"({length_ratio:.0%} of source); refusing unreviewed output."
        )

    source_numbers = _number_tokens(source)
    polished_numbers = _number_tokens(polished)
    if source_numbers != polished_numbers:
        raise PolishFidelityError(
            f"Polished chunk {index} changed numeric claims "
            f"(source={dict(source_numbers)}, output={dict(polished_numbers)}); "
            "use a stronger model or polish this chunk manually."
        )

    source_terms = _content_tokens(source)
    if len(source_terms) >= 6:
        coverage = len(source_terms & _content_tokens(polished)) / len(source_terms)
        if coverage < 0.45:
            raise PolishFidelityError(
                f"Polished chunk {index} retained only {coverage:.0%} of source terms; "
                "the model may have summarized or invented content."
            )


def _read_glossary(path: Path | None) -> str:
    if path is None:
        return ""
    glossary = Path(path).expanduser().resolve()
    if not glossary.is_file():
        raise FileNotFoundError(str(glossary))
    return glossary.read_text(encoding="utf-8").strip()


def _resolve_endpoint(cfg: PolishConfig) -> tuple[str, str | None]:
    if cfg.backend == "ollama":
        base_url = cfg.base_url.strip() or "http://127.0.0.1:11434"
        if base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/")[:-3]
        if not _is_loopback_url(base_url):
            raise PolishError("The ollama backend must use a localhost/loopback URL.")
        return base_url, None

    if cfg.backend == "openai-compatible":
        base_url = cfg.base_url.strip()
        if not base_url:
            raise PolishError("--polish-base-url is required for openai-compatible polishing.")
        if not _is_loopback_url(base_url) and not cfg.allow_remote:
            raise PolishError(
                "Remote AI polishing is disabled. Pass --allow-online-polish only after "
                "confirming that transcript text may be sent to that endpoint."
            )
        api_key = os.environ.get(cfg.api_key_env) if cfg.api_key_env else None
        return base_url, api_key

    raise PolishError(f"Unsupported polish backend: {cfg.backend}")


def _polish_chunk(
    chunk: str,
    *,
    index: int,
    total: int,
    glossary: str,
    instructions: str,
    cfg: PolishConfig,
) -> str:
    base_url, api_key = _resolve_endpoint(cfg)
    url = (
        f"{base_url.rstrip('/')}/api/chat"
        if cfg.backend == "ollama"
        else f"{base_url.rstrip('/')}/chat/completions"
    )
    glossary_note = (
        f"\n\nTerminology glossary (preserve these spellings and meanings):\n{glossary}"
        if glossary
        else ""
    )
    instructions_note = (
        f"\n\nAdditional user requirements (apply when they do not invent or alter facts):\n{instructions}"
        if instructions
        else ""
    )
    user_prompt = (
        f"Polish transcript chunk {index} of {total}. It is one continuous document; "
        "do not add an introduction or conclusion."
        f"{glossary_note}{instructions_note}\n\nTRANSCRIPT CHUNK:\n{chunk}"
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if cfg.backend == "ollama":
        options: dict[str, float | int] = {
            "temperature": 0.1,
            "num_predict": max(512, min(8192, len(chunk))),
        }
        if cfg.ollama_num_gpu is not None:
            options["num_gpu"] = cfg.ollama_num_gpu
        payload = {
            "model": cfg.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": options,
        }
    else:
        payload = {
            "model": cfg.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": max(512, min(8192, len(chunk))),
            "stream": False,
        }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=cfg.timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise PolishError(f"Polish endpoint returned HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise PolishError(f"Could not reach polish endpoint {url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PolishError("Polish endpoint returned invalid JSON.") from exc

    try:
        content = (
            body["message"]["content"]
            if cfg.backend == "ollama"
            else body["choices"][0]["message"]["content"]
        )
    except (KeyError, IndexError, TypeError) as exc:
        raise PolishError("Polish endpoint response did not contain assistant text.") from exc
    result = _strip_model_wrapping(str(content))
    if not result:
        raise PolishError(f"Polish endpoint returned empty text for chunk {index}.")
    _validate_polished_chunk(chunk, result, index)
    return result


def polish_transcript(text: str, cfg: PolishConfig) -> str:
    if cfg.backend == "none":
        return text
    if not cfg.model.strip():
        raise PolishError("--polish-model is required when AI polishing is enabled.")
    if cfg.chunk_chars < 500:
        raise ValueError("polish chunk size must be at least 500 characters")

    glossary = _read_glossary(cfg.glossary_path)
    instructions = _read_glossary(cfg.instructions_path)
    chunks = split_for_polish(text, cfg.chunk_chars)
    checkpoint_dir = Path(cfg.checkpoint_dir) if cfg.checkpoint_dir else None
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    polished: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        checkpoint_path: Path | None = None
        if checkpoint_dir:
            fingerprint = sha256(
                json.dumps(
                    {
                        "backend": cfg.backend,
                        "model": cfg.model,
                        "base_url": cfg.base_url,
                        "system_prompt": _SYSTEM_PROMPT,
                        "glossary": glossary,
                        "instructions": instructions,
                        "chunk": chunk,
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()[:16]
            checkpoint_path = checkpoint_dir / f"{index:04d}-{fingerprint}.txt"
            if checkpoint_path.is_file():
                cached = checkpoint_path.read_text(encoding="utf-8").strip()
                try:
                    _validate_polished_chunk(chunk, cached, index)
                except PolishFidelityError:
                    cached = chunk
                    checkpoint_path.write_text(cached + "\n", encoding="utf-8")
                polished.append(cached)
                continue

        try:
            result = _polish_chunk(
                chunk,
                index=index,
                total=len(chunks),
                glossary=glossary,
                instructions=instructions,
                cfg=cfg,
            )
        except PolishFidelityError:
            # Preserve the cleaned source rather than aborting a long job or
            # accepting a rewrite that altered its content.
            result = chunk
        if checkpoint_path:
            checkpoint_path.write_text(result + "\n", encoding="utf-8")
        polished.append(result)
    return "\n\n".join(polished).strip()
