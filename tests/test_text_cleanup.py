import sys
from types import ModuleType, SimpleNamespace

import pytest

from tts_from_youtube.text import basic_cleanup, load_text_input


def test_basic_cleanup_preserves_paragraph_breaks_when_enabled() -> None:
    raw = "Hello there.\nHow are you?\n\nI am fine.\n\n\nThis is last."
    cleaned = basic_cleanup(raw, preserve_paragraph_breaks=True)
    assert cleaned == "Hello there. How are you?\n\nI am fine.\n\nThis is last."


def test_basic_cleanup_flattens_breaks_by_default() -> None:
    raw = "Hello there.\nHow are you?\n\nI am fine."
    cleaned = basic_cleanup(raw)
    assert cleaned == "Hello there. How are you? I am fine."


def _mock_pypdf(monkeypatch: pytest.MonkeyPatch, pages: list[str]) -> None:
    fake_pypdf = ModuleType("pypdf")
    fake_pypdf.PdfReader = lambda path: SimpleNamespace(
        is_encrypted=False,
        pages=[SimpleNamespace(extract_text=lambda text=text: text) for text in pages],
    )
    fake_errors = ModuleType("pypdf.errors")
    fake_errors.FileNotDecryptedError = type("FileNotDecryptedError", (Exception,), {})
    fake_errors.PdfReadError = type("PdfReadError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)
    monkeypatch.setitem(sys.modules, "pypdf.errors", fake_errors)


def test_load_pdf_preserves_page_boundaries(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = tmp_path / "document.pdf"
    pdf.write_bytes(b"%PDF-test")
    _mock_pypdf(monkeypatch, ["First page.", "Second page."])

    assert load_text_input(pdf) == "First page.\n\nSecond page."


def test_load_pdf_rejects_image_only_document(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-test")
    _mock_pypdf(monkeypatch, [""])

    with pytest.raises(ValueError, match="OCR"):
        load_text_input(pdf)
