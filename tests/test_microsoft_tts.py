from tts_from_youtube.tts.microsoft_tts import split_text_for_microsoft


def test_split_text_for_microsoft_breaks_long_text_into_multiple_chunks() -> None:
    text = ("第一句。这是第二句！这是第三句？" * 400) + "\n\n" + ("Next sentence. " * 300)

    chunks = split_text_for_microsoft(text, max_chars=3000)

    assert len(chunks) > 1
    assert all(len(chunk) <= 3000 for chunk in chunks)
    assert "".join(chunks).replace("\n", "").replace(" ", "") == text.replace("\n", "").replace(" ", "")


def test_split_text_for_microsoft_handles_single_long_run_without_punctuation() -> None:
    text = "甲" * 6500

    chunks = split_text_for_microsoft(text, max_chars=3000)

    assert [len(chunk) for chunk in chunks] == [3000, 3000, 500]
