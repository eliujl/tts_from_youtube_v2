from __future__ import annotations

import json

import pytest

from tts_from_youtube.pipeline import RunConfig, run_local_file
from tts_from_youtube.polish import (
    PolishConfig,
    PolishError,
    _validate_polished_chunk,
    polish_transcript,
    split_for_polish,
)


def test_split_for_polish_respects_limit_and_content() -> None:
    text = ("First sentence. Second sentence. " * 30) + "\n\nFinal paragraph."
    chunks = split_for_polish(text, 500)

    assert len(chunks) > 1
    assert all(len(chunk) <= 500 for chunk in chunks)
    assert "".join("".join(chunks).split()) == "".join(text.split())


def test_ollama_must_be_local() -> None:
    cfg = PolishConfig(
        backend="ollama",
        model="test-model",
        base_url="https://example.com/v1",
    )

    with pytest.raises(PolishError, match="localhost"):
        polish_transcript("Example text.", cfg)


def test_remote_endpoint_requires_explicit_permission() -> None:
    cfg = PolishConfig(
        backend="openai-compatible",
        model="test-model",
        base_url="https://example.com/v1",
    )

    with pytest.raises(PolishError, match="allow-online-polish"):
        polish_transcript("Example text.", cfg)


def test_validation_rejects_changed_numeric_claim() -> None:
    source = "How long should a three month job take? Three months or less."
    changed = "A three to six month job is normal."

    with pytest.raises(PolishError, match="numeric claims"):
        _validate_polished_chunk(source, changed, 1)


def test_validation_rejects_summary_with_low_source_coverage() -> None:
    source = (
        "Release approval and control feelings while allowing fear and resistance "
        "to arise naturally during the workshop practice."
    )
    summary = "This passage offers general spiritual advice for students."

    with pytest.raises(PolishError, match="source terms"):
        _validate_polished_chunk(source, summary, 1)


def test_local_polish_returns_assistant_text(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {"message": {"content": "```text\nPolished text.\n```"}}
            ).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode())
        return FakeResponse()

    monkeypatch.setattr("tts_from_youtube.polish.urllib.request.urlopen", fake_urlopen)
    cfg = PolishConfig(backend="ollama", model="local-model", ollama_num_gpu=0)

    result = polish_transcript("Messy text.", cfg)

    assert result == "Polished text."
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["timeout"] == 600
    assert captured["payload"]["model"] == "local-model"
    assert captured["payload"]["think"] is False
    assert captured["payload"]["options"]["num_gpu"] == 0


def test_text_pipeline_writes_separate_tts_transcript(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "lecture.txt"
    source.write_text("Um, this is the raw lecture.", encoding="utf-8")
    monkeypatch.setattr(
        "tts_from_youtube.pipeline.polish_transcript",
        lambda text, cfg: "This is the polished lecture.",
    )
    cfg = RunConfig(
        out_dir=tmp_path / "out",
        tts_backend="none",
        polish_backend="ollama",
        polish_model="local-model",
    )

    output_dir = run_local_file(source, cfg)

    assert (output_dir / "lecture.transcript.txt").read_text(encoding="utf-8").startswith("Um")
    assert "raw lecture" in (output_dir / "lecture.transcript_clean.txt").read_text(
        encoding="utf-8"
    )
    assert (output_dir / "lecture.transcript_tts.txt").read_text(
        encoding="utf-8"
    ) == "This is the polished lecture.\n"
    manifest = json.loads((output_dir / "lecture.manifest.json").read_text(encoding="utf-8"))
    assert manifest["polish"]["backend"] == "ollama"
    assert manifest["artifacts"]["transcript_tts"].endswith("lecture.transcript_tts.txt")


def test_polish_checkpoints_resume_completed_chunks(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    source = "The original transcript remains faithful and complete."

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self) -> bytes:
            return json.dumps({"message": {"content": source}}).encode()

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        return FakeResponse()

    monkeypatch.setattr("tts_from_youtube.polish.urllib.request.urlopen", fake_urlopen)
    cfg = PolishConfig(
        backend="ollama",
        model="local-model",
        checkpoint_dir=tmp_path / "chunks",
    )

    assert polish_transcript(source, cfg) == source
    assert polish_transcript(source, cfg) == source
    assert calls == 1


def test_polish_preserves_chunk_when_model_fails_fidelity_check(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = "How long should a three month job take? Three months or less."

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {"message": {"content": "A three to six month job is normal."}}
            ).encode()

    monkeypatch.setattr(
        "tts_from_youtube.polish.urllib.request.urlopen",
        lambda request, timeout: FakeResponse(),
    )
    checkpoint_dir = tmp_path / "chunks"
    cfg = PolishConfig(
        backend="ollama",
        model="local-model",
        checkpoint_dir=checkpoint_dir,
    )

    assert polish_transcript(source, cfg) == source
    assert next(checkpoint_dir.glob("*.txt")).read_text(encoding="utf-8").strip() == source
