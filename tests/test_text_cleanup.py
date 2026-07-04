from tts_from_youtube.text import basic_cleanup


def test_basic_cleanup_preserves_paragraph_breaks_when_enabled() -> None:
    raw = "Hello there.\nHow are you?\n\nI am fine.\n\n\nThis is last."
    cleaned = basic_cleanup(raw, preserve_paragraph_breaks=True)
    assert cleaned == "Hello there. How are you?\n\nI am fine.\n\nThis is last."


def test_basic_cleanup_flattens_breaks_by_default() -> None:
    raw = "Hello there.\nHow are you?\n\nI am fine."
    cleaned = basic_cleanup(raw)
    assert cleaned == "Hello there. How are you? I am fine."
