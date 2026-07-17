from __future__ import annotations

import json
import sys
from email.message import Message
from http.cookiejar import CookieJar
from types import ModuleType, SimpleNamespace
from urllib.error import URLError

import pytest
from typer.testing import CliRunner

from tts_from_youtube.cli import app
from tts_from_youtube.pipeline import RunConfig, run_webpage
from tts_from_youtube.ui import _run as ui_run
from tts_from_youtube.ui import _webpage_error_message
from tts_from_youtube.webpage import (
    MAX_WEBPAGE_BYTES,
    WebpageExtractionError,
    WebpageText,
    extract_webpage_text,
)


class FakeResponse:
    def __init__(
        self,
        body: bytes,
        *,
        url: str = "https://example.org/final",
        content_type: str | None = "text/html; charset=utf-8",
        status: int = 200,
    ) -> None:
        self._body = body
        self._url = url
        self.status = status
        self.headers = Message()
        if content_type is not None:
            self.headers["Content-Type"] = content_type
        self.read_size: int | None = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def geturl(self) -> str:
        return self._url

    def read(self, size: int = -1) -> bytes:
        self.read_size = size
        if size < 0:
            return self._body
        return self._body[:size]


def _install_fake_trafilatura(monkeypatch: pytest.MonkeyPatch, text: str | None = None) -> None:
    fake = ModuleType("trafilatura")

    def extract(source_html: str, **kwargs):
        if text is not None:
            return text
        return (
            "Important Article Heading\n\n"
            "This is the first main paragraph with enough readable substance for extraction.\n\n"
            "This is the second main paragraph that should remain while navigation disappears."
        )

    def extract_metadata(source_html: str, default_url: str):
        return SimpleNamespace(title="Important Article")

    fake.extract = extract
    fake.extract_metadata = extract_metadata
    monkeypatch.setitem(sys.modules, "trafilatura", fake)


def test_extract_article_records_title_and_excludes_page_chrome(monkeypatch: pytest.MonkeyPatch) -> None:
    html = b"""
    <html><head><title>Fallback Title</title><script>alert(1)</script></head>
    <body><nav>Home Subscribe</nav><main>
    <h1>Important Article Heading</h1>
    <p>This is the first main paragraph with enough readable substance for extraction.</p>
    <p>This is the second main paragraph that should remain while navigation disappears.</p>
    </main><footer>Copyright footer</footer><!-- comment --></body></html>
    """
    _install_fake_trafilatura(monkeypatch)
    response = FakeResponse(html)
    monkeypatch.setattr("tts_from_youtube.webpage._open_url", lambda url, **kwargs: response)

    page = extract_webpage_text("https://example.org/article")

    assert page.title == "Important Article"
    assert page.final_url == "https://example.org/final"
    assert "first main paragraph" in page.text
    assert "Home Subscribe" not in page.text
    assert "Copyright footer" not in page.text
    assert "comment" not in page.text


def test_redirected_final_url_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_trafilatura(monkeypatch)
    monkeypatch.setattr(
        "tts_from_youtube.webpage._open_url",
        lambda url, **kwargs: FakeResponse(b"<html><title>T</title></html>", url="https://news.example/final"),
    )

    page = extract_webpage_text("https://news.example/start")

    assert page.url == "https://news.example/start"
    assert page.final_url == "https://news.example/final"


@pytest.mark.parametrize("url", ["file:///tmp/a.html", "ftp://example.org/a"])
def test_non_http_schemes_are_rejected(url: str) -> None:
    with pytest.raises(WebpageExtractionError, match="http"):
        extract_webpage_text(url)


def test_credential_urls_are_rejected() -> None:
    with pytest.raises(WebpageExtractionError, match="username or password"):
        extract_webpage_text("https://user:secret@example.org/article")


def test_non_html_content_type_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tts_from_youtube.webpage._open_url",
        lambda url, **kwargs: FakeResponse(b"%PDF", content_type="application/pdf"),
    )

    with pytest.raises(WebpageExtractionError, match="not HTML"):
        extract_webpage_text("https://example.org/file.pdf")


def test_oversized_response_is_rejected_with_bounded_read(monkeypatch: pytest.MonkeyPatch) -> None:
    response = FakeResponse(b"x" * (MAX_WEBPAGE_BYTES + 1))
    monkeypatch.setattr("tts_from_youtube.webpage._open_url", lambda url, **kwargs: response)

    with pytest.raises(WebpageExtractionError, match="10 MiB"):
        extract_webpage_text("https://example.org/huge")

    assert response.read_size == MAX_WEBPAGE_BYTES + 1


def test_short_extraction_gets_actionable_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_trafilatura(monkeypatch, text="Home")
    monkeypatch.setattr(
        "tts_from_youtube.webpage._open_url",
        lambda url, **kwargs: FakeResponse(b"<html><body>Home</body></html>"),
    )

    with pytest.raises(WebpageExtractionError, match=r"Save or upload.*\.txt"):
        extract_webpage_text("https://example.org/nav")


def test_network_failure_becomes_stable_extraction_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(url: str, **kwargs):
        raise URLError("offline")

    monkeypatch.setattr("tts_from_youtube.webpage._open_url", fail)

    with pytest.raises(WebpageExtractionError, match="Could not fetch webpage"):
        extract_webpage_text("https://example.org/article")


def test_browser_cookie_option_is_passed_to_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    payload = [
        {"data": {"children": [{"kind": "t3", "data": {"title": "Post", "author": "op", "subreddit": "example", "selftext": "Opening post"}}]}},
        {"data": {"children": []}},
    ]

    def fake_open(url: str, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse(json.dumps(payload).encode(), content_type="application/json")

    monkeypatch.setattr("tts_from_youtube.webpage._open_url", fake_open)

    extract_webpage_text(
        "https://www.reddit.com/r/example/comments/abc/thread/",
        browser_cookies_from="chrome",
        browser_profile="Default",
    )

    assert captured["browser_cookies_from"] == "chrome"
    assert captured["browser_profile"] == "Default"
    assert ".json?" in str(captured["url"])
    assert captured["accept"] == "application/json"


def test_reddit_post_includes_nested_replies_and_reports_omissions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = [
        {
            "data": {
                "children": [
                    {
                        "kind": "t3",
                        "data": {
                            "title": "A useful discussion",
                            "author": "original_poster",
                            "subreddit": "example",
                            "selftext": "This is the opening post.",
                        },
                    }
                ]
            }
        },
        {
            "data": {
                "children": [
                    {
                        "kind": "t1",
                        "data": {
                            "author": "first_commenter",
                            "body": "This is a top-level comment.",
                            "replies": {
                                "data": {
                                    "children": [
                                        {
                                            "kind": "t1",
                                            "data": {
                                                "author": "reply_author",
                                                "body": "This is a nested reply.",
                                                "replies": "",
                                            },
                                        },
                                        {"kind": "more", "data": {"count": 3, "children": ["a", "b", "c"]}},
                                    ]
                                }
                            },
                        },
                    }
                ]
            }
        },
    ]
    monkeypatch.setattr(
        "tts_from_youtube.webpage._open_url",
        lambda url, **kwargs: FakeResponse(json.dumps(payload).encode(), content_type="application/json"),
    )

    page = extract_webpage_text("https://www.reddit.com/r/example/comments/abc/thread/")

    assert page.title == "A useful discussion"
    assert "This is the opening post." in page.text
    assert "Comment by u/first_commenter" in page.text
    assert "Reply, level 1 by u/reply_author" in page.text
    assert "omitted 3 additional comments" in page.text


def test_cookie_file_is_passed_to_fetch(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_trafilatura(monkeypatch)
    captured: dict[str, object] = {}
    cookie_file = tmp_path / "cookies.txt"
    cookie_file.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")

    def fake_open(url: str, **kwargs):
        captured.update(kwargs)
        return FakeResponse(b"<html><title>Article</title></html>")

    monkeypatch.setattr("tts_from_youtube.webpage._open_url", fake_open)
    extract_webpage_text("https://example.org/article", cookie_file=cookie_file)

    assert captured["cookie_file"] == cookie_file


def test_load_cookie_file_accepts_netscape_format(tmp_path) -> None:
    from tts_from_youtube.webpage import _load_cookie_file

    cookie_file = tmp_path / "cookies.txt"
    cookie_file.write_text(
        "# Netscape HTTP Cookie File\n"
        ".reddit.com\tTRUE\t/\tTRUE\t2147483647\tsession\ttest-value\n",
        encoding="utf-8",
    )

    jar = _load_cookie_file(cookie_file)

    assert len(jar) == 1


def test_cookie_loader_wraps_browser_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = ModuleType("yt_dlp.cookies")
    fake.CookieLoadError = type("CookieLoadError", (Exception,), {})

    def fail(*args, **kwargs):
        raise fake.CookieLoadError("locked")

    fake.extract_cookies_from_browser = fail
    monkeypatch.setitem(sys.modules, "yt_dlp.cookies", fake)

    from tts_from_youtube.webpage import _load_browser_cookies

    with pytest.raises(WebpageExtractionError, match="Could not load cookies"):
        _load_browser_cookies("chrome", None)


def test_cookie_loader_returns_cookiejar(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = ModuleType("yt_dlp.cookies")
    fake.CookieLoadError = type("CookieLoadError", (Exception,), {})
    jar = CookieJar()
    fake.extract_cookies_from_browser = lambda browser_name, profile=None: jar
    monkeypatch.setitem(sys.modules, "yt_dlp.cookies", fake)

    from tts_from_youtube.webpage import _load_browser_cookies

    assert _load_browser_cookies("chrome", "Default") is jar


def test_run_webpage_writes_text_artifacts_and_manifest_without_asr(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = WebpageText(
        url="https://example.org/start",
        final_url="https://example.org/final",
        title="Article Title",
        text="I mean this article is like a careful authored essay with enough text to use.",
    )
    monkeypatch.setattr("tts_from_youtube.pipeline.extract_webpage_text", lambda url, **kwargs: page)

    def fail_asr(*args, **kwargs):
        raise AssertionError("ASR should not run for webpages")

    monkeypatch.setattr("tts_from_youtube.pipeline._transcribe", fail_asr)
    cfg = RunConfig(out_dir=tmp_path / "out", tts_backend="none")

    out_dir = run_webpage("https://example.org/start", cfg)

    assert (out_dir / "Article Title.transcript.txt").exists()
    clean = (out_dir / "Article Title.transcript_clean.txt").read_text(encoding="utf-8")
    assert "I mean" in clean
    assert " like " in clean
    manifest = json.loads((out_dir / "Article Title.manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_kind"] == "webpage"
    assert manifest["source_url"] == "https://example.org/start"
    assert manifest["source_final_url"] == "https://example.org/final"
    assert "asr" not in manifest
    assert manifest["artifacts"]["transcript_clean"].endswith("Article Title.transcript_clean.txt")


def test_run_webpage_polishes_and_synthesizes_polished_text(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = WebpageText(
        url="https://example.org/article",
        final_url="https://example.org/article",
        title="Polish Me",
        text="Original article text with enough words for a realistic article extraction.",
    )
    captured: dict[str, str] = {}
    monkeypatch.setattr("tts_from_youtube.pipeline.extract_webpage_text", lambda url, **kwargs: page)
    monkeypatch.setattr("tts_from_youtube.pipeline.polish_transcript", lambda text, cfg: "Polished article text.")

    def fake_synthesize(text: str, out_dir, cfg):
        captured["text"] = text
        output = out_dir / "fake.wav"
        output.write_text("audio", encoding="utf-8")
        return output

    monkeypatch.setattr("tts_from_youtube.pipeline._synthesize", fake_synthesize)
    cfg = RunConfig(
        out_dir=tmp_path / "out",
        tts_backend="piper",
        polish_backend="ollama",
        polish_model="local-model",
    )

    out_dir = run_webpage("https://example.org/article", cfg)

    assert captured["text"] == "Polished article text."
    assert (out_dir / "Polish Me.transcript_tts.txt").read_text(encoding="utf-8") == "Polished article text.\n"
    manifest = json.loads((out_dir / "Polish Me.manifest.json").read_text(encoding="utf-8"))
    assert manifest["polish"]["backend"] == "ollama"
    assert manifest["tts"]["backend"] == "piper"
    assert manifest["artifacts"]["transcript_tts"].endswith("Polish Me.transcript_tts.txt")


def test_cli_webpage_help_exits_successfully() -> None:
    result = CliRunner().invoke(app, ["webpage", "--help"])

    assert result.exit_code == 0
    assert "web-timeout" in result.stdout


def test_cli_run_rejects_reddit_with_webpage_hint() -> None:
    result = CliRunner().invoke(app, ["run", "https://www.reddit.com/r/example/comments/abc/thread/"])

    assert result.exit_code == 2
    assert "y2tts webpage" in result.stdout
    assert "--web-cookies-from-browser chrome" in result.stdout


def test_ui_download_only_rejects_reddit_with_webpage_hint(tmp_path) -> None:
    with pytest.raises(Exception, match="Choose Source = Webpage"):
        ui_run(
            "YouTube",
            "https://www.reddit.com/r/example/comments/abc/thread/",
            None,
            str(tmp_path),
            "Download only (YouTube)",
            "distil-large-v3",
            "auto",
            "",
            "",
            True,
            False,
            "none",
            "",
            "",
            "OPENAI_API_KEY",
            None,
            None,
            8000,
            600,
            0,
            False,
            "none",
            1.0,
            False,
            True,
            "en_US-lessac-medium",
            "",
            False,
            "en-US-MichelleNeural",
            "tts_models/en/jenny/jenny",
            None,
            "",
            "video",
            "wav",
            30,
            "",
            "",
            None,
        )


def test_ui_youtube_source_auto_routes_reddit_to_webpage(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_run_webpage(url, cfg, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        out_dir = tmp_path / "AutoWebpage"
        out_dir.mkdir()
        (out_dir / "AutoWebpage.transcript_clean.txt").write_text("preview", encoding="utf-8")
        return out_dir

    monkeypatch.setattr("tts_from_youtube.ui.run_webpage", fake_run_webpage)

    ui_run(
        "YouTube",
        "https://www.reddit.com/r/example/comments/abc/thread/",
        None,
        str(tmp_path),
        "Run (ASR + TTS)",
        "distil-large-v3",
        "auto",
        "",
        "",
        True,
        False,
        "none",
        "",
        "",
        "OPENAI_API_KEY",
        None,
        None,
        8000,
        600,
        0,
        False,
        "none",
        1.0,
        False,
        True,
        "en_US-lessac-medium",
        "",
        False,
        "en-US-MichelleNeural",
        "tts_models/en/jenny/jenny",
        None,
        "",
        "video",
        "wav",
        30,
        "chrome",
        "Default",
        None,
    )

    assert captured["url"].startswith("https://www.reddit.com/")
    assert captured["browser_cookies_from"] == "chrome"


def test_ui_webpage_passes_browser_cookie_options(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_run_webpage(
        url,
        cfg,
        *,
        timeout_seconds=30,
        browser_cookies_from=None,
        browser_profile=None,
        cookie_file=None,
    ):
        captured["url"] = url
        captured["browser_cookies_from"] = browser_cookies_from
        captured["browser_profile"] = browser_profile
        captured["cookie_file"] = cookie_file
        out_dir = tmp_path / "Fake"
        out_dir.mkdir()
        (out_dir / "Fake.transcript_clean.txt").write_text("preview", encoding="utf-8")
        return out_dir

    monkeypatch.setattr("tts_from_youtube.ui.run_webpage", fake_run_webpage)

    ui_run(
        "Webpage",
        "https://www.reddit.com/r/example/comments/abc/thread/",
        None,
        str(tmp_path),
        "Run (ASR + TTS)",
        "distil-large-v3",
        "auto",
        "",
        "",
        True,
        False,
        "none",
        "",
        "",
        "OPENAI_API_KEY",
        None,
        None,
        8000,
        600,
        0,
        False,
        "none",
        1.0,
        False,
        True,
        "en_US-lessac-medium",
        "",
        False,
        "en-US-MichelleNeural",
        "tts_models/en/jenny/jenny",
        None,
        "",
        "video",
        "wav",
        30,
        "chrome",
        "Default",
        None,
    )

    assert captured["url"].startswith("https://www.reddit.com/")
    assert captured["browser_cookies_from"] == "chrome"
    assert captured["browser_profile"] == "Default"
    assert captured["cookie_file"] is None


def test_ui_webpage_error_message_explains_cookie_permission_failure() -> None:
    message = _webpage_error_message(
        PermissionError("Permission denied: Brave-Browser User Data Default Network Cookies"),
        url="https://www.reddit.com/r/example/comments/abc/thread/",
        browser_cookies_from="brave",
        browser_profile="Default",
    )

    assert "Webpage extraction failed" in message
    assert "Close brave completely" in message
    assert "Current profile setting: Default" in message
    assert "save the page text" in message


def test_ui_webpage_error_message_explains_dpapi_failure() -> None:
    message = _webpage_error_message(
        RuntimeError("Failed to decrypt with DPAPI"),
        url="https://www.reddit.com/r/example/comments/abc/thread/",
        browser_cookies_from="brave",
        browser_profile="Default",
    )

    assert "cookies.txt" in message
    assert "Firefox" in message
    assert "profile path" in message
