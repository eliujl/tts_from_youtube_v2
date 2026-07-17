from __future__ import annotations

import html
import ipaddress
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from email.message import Message
from http.client import HTTPResponse
from http.cookiejar import CookieJar, LoadError, MozillaCookieJar
from pathlib import Path
from typing import BinaryIO

MAX_WEBPAGE_BYTES = 10 * 1024 * 1024
USER_AGENT = "y2tts/0.2 (+https://github.com/local/y2tts; webpage-to-tts)"


class WebpageExtractionError(ValueError):
    """User-facing webpage extraction failure."""


@dataclass(frozen=True)
class WebpageText:
    url: str
    final_url: str
    title: str
    text: str


def _validate_url(url: str) -> urllib.parse.ParseResult:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise WebpageExtractionError("Webpage URL must start with http:// or https://.")
    if not parsed.hostname:
        raise WebpageExtractionError("Webpage URL must include a hostname.")
    if parsed.username or parsed.password:
        raise WebpageExtractionError("Webpage URL must not include a username or password.")

    try:
        ip = ipaddress.ip_address(parsed.hostname.strip("[]"))
    except ValueError:
        return parsed

    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_unspecified
    ):
        raise WebpageExtractionError("Webpage URL must not use a private or local IP address.")
    return parsed


def _is_reddit_post_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    return host in {"reddit.com", "www.reddit.com", "old.reddit.com", "new.reddit.com"} and bool(
        re.search(r"/(?:r/[^/]+/)?comments/[^/]+", parsed.path, flags=re.IGNORECASE)
    )


def _reddit_json_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.rstrip("/")
    if not path.endswith(".json"):
        path += ".json"
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query.extend(
        [
            ("raw_json", "1"),
            ("limit", "500"),
            ("depth", "20"),
            ("sort", "top"),
        ]
    )
    return urllib.parse.urlunparse(("https", "www.reddit.com", path, "", urllib.parse.urlencode(query), ""))


def _reddit_listing_children(listing: object) -> list[dict]:
    if not isinstance(listing, dict):
        return []
    data = listing.get("data")
    if not isinstance(data, dict):
        return []
    children = data.get("children")
    return [child for child in children if isinstance(child, dict)] if isinstance(children, list) else []


def _format_reddit_thread(payload: object, source_url: str) -> WebpageText:
    if not isinstance(payload, list) or len(payload) < 2:
        raise WebpageExtractionError("Reddit returned an unexpected thread response.")

    post_children = _reddit_listing_children(payload[0])
    if not post_children or not isinstance(post_children[0].get("data"), dict):
        raise WebpageExtractionError("Reddit did not return the original post.")
    post = post_children[0]["data"]
    title = str(post.get("title") or "Reddit post").strip()
    author = str(post.get("author") or "unknown")
    subreddit = str(post.get("subreddit") or "unknown")
    selftext = str(post.get("selftext") or "").strip()
    lines = [title, f"Post by u/{author} in r/{subreddit}"]
    if selftext:
        lines.extend(["", selftext])
    elif post.get("url"):
        lines.extend(["", f"Link: {post['url']}"])

    comment_count = 0
    omitted_count = 0

    def append_comments(children: list[dict], depth: int = 0) -> None:
        nonlocal comment_count, omitted_count
        for child in children:
            kind = child.get("kind")
            data = child.get("data")
            if not isinstance(data, dict):
                continue
            if kind == "more":
                omitted_count += int(data.get("count") or len(data.get("children") or []))
                continue
            if kind != "t1":
                continue
            body = str(data.get("body") or "").strip()
            if body:
                comment_count += 1
                label = "Comment" if depth == 0 else f"Reply, level {depth}"
                lines.extend(["", f"{label} by u/{data.get('author') or 'unknown'}:", body])
            replies = data.get("replies")
            if isinstance(replies, dict):
                append_comments(_reddit_listing_children(replies), depth + 1)

    append_comments(_reddit_listing_children(payload[1]))
    if omitted_count:
        lines.extend(
            [
                "",
                f"Reddit omitted {omitted_count} additional comments from this response.",
            ]
        )
    if not comment_count:
        lines.extend(["", "No replies were returned by Reddit."])

    text = _normalize_extracted_text("\n".join(lines))
    return WebpageText(url=source_url, final_url=source_url, title=title, text=text)


def _load_browser_cookies(browser_name: str, profile: str | None) -> CookieJar:
    try:
        from yt_dlp.cookies import CookieLoadError, extract_cookies_from_browser
        from yt_dlp.utils import DownloadError
    except ImportError as exc:  # pragma: no cover - yt-dlp is a normal dependency
        raise WebpageExtractionError("Browser cookie loading requires yt-dlp.") from exc

    try:
        return extract_cookies_from_browser(browser_name, profile=profile or None)
    except (CookieLoadError, DownloadError, OSError, PermissionError) as exc:
        raise WebpageExtractionError(
            f"Could not load cookies from {browser_name}. Make sure you are logged in, "
            "then close that browser completely and retry. If the browser is still open, "
            "Windows may lock its cookie database. If you use a non-default profile, set "
            "the matching browser profile. Original cookie error: "
            f"{exc}"
        ) from exc
    except ValueError as exc:
        raise WebpageExtractionError(
            "Unknown browser for cookie loading. Use chrome, edge, firefox, brave, chromium, "
            "opera, vivaldi, or safari."
        ) from exc


def _load_cookie_file(cookie_file: Path) -> CookieJar:
    path = cookie_file.expanduser().resolve()
    if not path.is_file():
        raise WebpageExtractionError(f"Cookie file was not found: {path}")

    jar = MozillaCookieJar(str(path))
    try:
        jar.load(ignore_discard=True, ignore_expires=True)
    except (LoadError, OSError) as exc:
        raise WebpageExtractionError(
            "Could not read the cookie file. Export it in Netscape cookies.txt format; "
            "its first line should be '# Netscape HTTP Cookie File'."
        ) from exc
    return jar


def _open_url(
    url: str,
    *,
    timeout_seconds: int,
    browser_cookies_from: str | None = None,
    browser_profile: str | None = None,
    cookie_file: Path | None = None,
    accept: str = "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
) -> HTTPResponse:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": accept,
        },
    )
    if browser_cookies_from and cookie_file:
        raise WebpageExtractionError("Choose either browser cookies or a cookies.txt file, not both.")
    if cookie_file or browser_cookies_from:
        cookiejar = (
            _load_cookie_file(cookie_file)
            if cookie_file
            else _load_browser_cookies(browser_cookies_from or "", browser_profile)
        )
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookiejar))
        return opener.open(request, timeout=timeout_seconds)
    return urllib.request.urlopen(request, timeout=timeout_seconds)


def _content_type(headers: Message) -> str:
    return (headers.get_content_type() or "").lower()


def _reject_non_html(headers: Message) -> None:
    content_type = _content_type(headers)
    if not content_type:
        return
    if content_type in {"text/html", "application/xhtml+xml"}:
        return
    if content_type.startswith("text/") and content_type not in {"text/css", "text/csv"}:
        return
    raise WebpageExtractionError(
        f"Webpage response was not HTML ({content_type}). Save or upload the article as .txt instead."
    )


def _read_limited(response: BinaryIO) -> bytes:
    data = response.read(MAX_WEBPAGE_BYTES + 1)
    if len(data) > MAX_WEBPAGE_BYTES:
        raise WebpageExtractionError("Webpage is larger than 10 MiB. Save or upload the article as .txt instead.")
    return data


def _decode_html(data: bytes, headers: Message) -> str:
    charset = headers.get_content_charset()
    try:
        if charset:
            return data.decode(charset, errors="replace")
    except LookupError:
        pass
    return data.decode("utf-8", errors="replace")


def _html_title(source_html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", source_html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return html.unescape(re.sub(r"\s+", " ", match.group(1))).strip()


def _normalize_extracted_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.splitlines()]
    normalized = "\n".join(lines).strip()
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized


def _fallback_error() -> WebpageExtractionError:
    return WebpageExtractionError(
        "Could not extract enough readable article text. The page may be blocked, paywalled, "
        "JavaScript-only, or mostly navigation. Save or upload the article as .txt instead."
    )


def _extract_reddit_post(
    url: str,
    *,
    timeout_seconds: int,
    browser_cookies_from: str | None,
    browser_profile: str | None,
    cookie_file: Path | None,
) -> WebpageText:
    try:
        with _open_url(
            _reddit_json_url(url),
            timeout_seconds=timeout_seconds,
            browser_cookies_from=browser_cookies_from,
            browser_profile=browser_profile,
            cookie_file=cookie_file,
            accept="application/json",
        ) as response:
            status = getattr(response, "status", 200)
            if status >= 400:
                raise WebpageExtractionError(f"Reddit returned HTTP {status}.")
            payload = json.loads(_read_limited(response).decode("utf-8", errors="replace"))
    except WebpageExtractionError:
        raise
    except json.JSONDecodeError as exc:
        raise WebpageExtractionError(
            "Reddit did not return thread data. Check that the selected Firefox profile is logged in."
        ) from exc
    except urllib.error.HTTPError as exc:
        raise WebpageExtractionError(f"Reddit returned HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:
        raise WebpageExtractionError(f"Could not fetch Reddit thread: {exc.reason}") from exc
    except OSError as exc:
        raise WebpageExtractionError(f"Could not fetch Reddit thread: {exc}") from exc
    return _format_reddit_thread(payload, url)


def extract_webpage_text(
    url: str,
    *,
    timeout_seconds: int = 30,
    browser_cookies_from: str | None = None,
    browser_profile: str | None = None,
    cookie_file: Path | None = None,
) -> WebpageText:
    """Fetch a user-supplied URL and extract listenable article text."""
    _validate_url(url)
    if _is_reddit_post_url(url):
        return _extract_reddit_post(
            url,
            timeout_seconds=timeout_seconds,
            browser_cookies_from=browser_cookies_from,
            browser_profile=browser_profile,
            cookie_file=cookie_file,
        )
    try:
        with _open_url(
            url,
            timeout_seconds=timeout_seconds,
            browser_cookies_from=browser_cookies_from,
            browser_profile=browser_profile,
            cookie_file=cookie_file,
        ) as response:
            status = getattr(response, "status", 200)
            if status >= 400:
                raise WebpageExtractionError(f"Webpage returned HTTP {status}.")

            final_url = response.geturl()
            _validate_url(final_url)
            headers = response.headers
            _reject_non_html(headers)
            html_bytes = _read_limited(response)
            source_html = _decode_html(html_bytes, headers)
    except WebpageExtractionError:
        raise
    except urllib.error.HTTPError as exc:
        raise WebpageExtractionError(f"Webpage returned HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:
        raise WebpageExtractionError(f"Could not fetch webpage: {exc.reason}") from exc
    except OSError as exc:
        raise WebpageExtractionError(f"Could not fetch webpage: {exc}") from exc

    try:
        import trafilatura

        extracted = trafilatura.extract(
            source_html,
            url=final_url,
            output_format="txt",
            include_comments=False,
            include_tables=False,
            include_links=False,
            include_images=False,
            include_formatting=False,
            favor_precision=True,
            deduplicate=True,
        )
        metadata = trafilatura.extract_metadata(source_html, default_url=final_url)
    except Exception as exc:
        raise _fallback_error() from exc

    text = _normalize_extracted_text(extracted or "")
    if len(re.sub(r"\s+", "", text)) < 80:
        raise _fallback_error()

    parsed_final = urllib.parse.urlparse(final_url)
    title = (getattr(metadata, "title", "") if metadata else "") or _html_title(source_html)
    title = title.strip() or parsed_final.hostname or "webpage"
    return WebpageText(url=url, final_url=final_url, title=title, text=text)
