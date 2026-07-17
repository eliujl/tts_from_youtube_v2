# Implementation Guide: Webpage -> Text -> Polish -> TTS

## Audience and goal

This is an execution guide for a smaller coding model. Implement general webpage input without changing the existing YouTube behavior.

The finished program must accept an ordinary `http://` or `https://` article URL, extract its main readable text, save the original extracted text, apply the existing cleanup and optional AI-polish stages, and optionally synthesize it with the existing TTS backends.

Do not treat a general webpage URL as a YouTube URL. Do not run ASR for a webpage.

## Current code facts

- `src/tts_from_youtube/pipeline.py::run_local_file` already implements the desired text flow for `.txt`, `.md`, `.vtt`, and `.pdf`:
  1. load raw text;
  2. write `<title>.transcript.txt`;
  3. run `basic_cleanup` and write `<title>.transcript_clean.txt`;
  4. run `_prepare_tts_text`, which optionally creates `<title>.transcript_tts.txt`;
  5. run `_synthesize`;
  6. write the manifest.
- `src/tts_from_youtube/text.py::load_text_input` handles local text documents only.
- `src/tts_from_youtube/polish.py` already supplies chunking, checkpoints, local/remote controls, and fidelity validation. Reuse it unchanged unless a failing test proves a change is required.
- `src/tts_from_youtube/ui.py` currently has only `YouTube` and `Local file` source types.
- `src/tts_from_youtube/cli.py` has `run` for YouTube and `local` for local files. Add a dedicated webpage command; do not overload these with ambiguous URL guessing.

## Required design

### 1. Add a webpage extraction module

Create `src/tts_from_youtube/webpage.py` with these public types/functions:

```python
@dataclass(frozen=True)
class WebpageText:
    url: str
    final_url: str
    title: str
    text: str

def extract_webpage_text(url: str, *, timeout_seconds: int = 30) -> WebpageText:
    ...
```

Use a main-content extractor, not raw BeautifulSoup text. Add `trafilatura>=2.0.0,<3.0.0` to the normal project dependencies in `pyproject.toml`. Fetch the response in this module and pass the downloaded HTML to `trafilatura.extract` so request policy is controlled by this project.

Extraction requirements:

- Accept only `http` and `https` URLs.
- Reject a URL with missing hostname or embedded username/password.
- Set a normal, identifiable `User-Agent` and an `Accept` header favoring HTML.
- Use the supplied timeout.
- Limit the downloaded response to 10 MiB. Read at most `10 MiB + 1 byte`, then reject oversized content.
- After redirects, validate the final URL again.
- Accept HTML/XHTML content types. If the server omits `Content-Type`, allow extraction; reject a clearly non-HTML type such as audio, video, image, archive, or PDF.
- Decode using the response charset when available, with UTF-8 replacement fallback.
- Ask Trafilatura to preserve paragraphs and useful headings while excluding comments, navigation, tables, links, images, and formatting markup. Plain listenable text is the output.
- Extract a useful title from Trafilatura metadata or the HTML `<title>`. Fall back to the hostname. Sanitize the title later at the pipeline boundary.
- Normalize CRLF/CR to LF, trim line whitespace, collapse three or more blank lines to two, and reject empty or implausibly short extraction (less than 80 non-whitespace characters). The error must suggest saving/uploading the article as `.txt` when extraction is blocked or the page is JavaScript-only.
- Wrap network, HTTP, decoding, and extraction failures in a concise `ValueError` or a new `WebpageExtractionError`. Do not expose a traceback as the normal user-facing explanation.

Security note: at minimum, document that URLs are user-supplied and may access the network. Preferably reject loopback, link-local, multicast, unspecified, and private IP literals. Do not add DNS resolution based blocking in this first patch unless it is implemented without a time-of-check/time-of-use gap; a superficial DNS check gives false confidence. The UI is local, so state this limitation in the README.

### 2. Refactor the shared text pipeline

In `src/tts_from_youtube/pipeline.py`, remove duplication by extracting a helper:

```python
def _run_text(
    raw_text: str,
    title: str,
    cfg: RunConfig,
    *,
    source: dict[str, object],
) -> Path:
    ...
```

Behavior:

- Validate `tts_speed > 0` and non-empty `raw_text`.
- Sanitize `title`, create `cfg.out_dir / sanitized_title`, and use existing artifact naming helpers.
- Preserve the raw extracted/loaded text in `.transcript.txt` before cleanup.
- Run `basic_cleanup`, `_prepare_tts_text`, and `_synthesize` in the existing order.
- Write the existing polish/TTS/artifact manifest fields and merge in `source`.
- Keep the existing local document output names and semantics unchanged.
- Change the text branch of `run_local_file` to call `_run_text`.

Add:

```python
def run_webpage(url: str, cfg: RunConfig, *, timeout_seconds: int = 30) -> Path:
    page = extract_webpage_text(url, timeout_seconds=timeout_seconds)
    return _run_text(
        page.text,
        page.title,
        cfg,
        source={
            "source_url": page.url,
            "source_final_url": page.final_url,
            "input_kind": "webpage",
        },
    )
```

Do not write the fetched HTML as an artifact. It may contain scripts, tracking data, or private page state. The extracted raw text is the reviewable source artifact.

### 3. Avoid destructive cleanup of authored prose

`basic_cleanup` currently removes `like`, `you know`, and `I mean` by default. That can change the meaning or voice of an authored article. Make `RunConfig` carry `remove_fillers: bool = True` to preserve current audio-transcript behavior.

For webpage and local text/document inputs, call the shared helper with filler removal disabled by default. One clean way is to give `_run_text` a `remove_fillers: bool = False` argument and pass it to `basic_cleanup`. Leave ASR paths using `cfg.remove_fillers` so existing behavior is compatible.

Do not perform Markdown-to-plain-text conversion as part of this webpage patch. Existing `.md` behavior can be improved separately.

### 4. Add a dedicated CLI command

In `src/tts_from_youtube/cli.py`, add:

```text
y2tts webpage URL [the same polish and TTS options as `local`] [--web-timeout 30]
```

Implementation guidance:

- Factor construction of `RunConfig` only if that refactor stays small and obvious. Avoid a broad CLI rewrite.
- Default `--tts` to `none`, matching the safer UI review workflow. The first run should conveniently support extraction/polish-only review.
- Support all current polish settings, voices, speech rate, paragraph preservation, and MP3 output.
- Help text must say that fetching the webpage uses the network and that Microsoft TTS and a remote polish endpoint transmit the extracted text to external services.
- Call `run_webpage(url, cfg, timeout_seconds=web_timeout)` and print completion with the existing helper.

Examples to add to README:

```powershell
# Extract only, then review transcript_clean.txt
y2tts webpage "https://example.org/article" --out out --tts none

# Extract and polish locally, then review transcript_tts.txt
y2tts webpage "https://example.org/article" --out out --tts none `
  --polish ollama --polish-model MODEL_NAME

# After review, synthesize (this fetches/extracts the page again)
y2tts webpage "https://example.org/article" --out out --tts microsoft --mp3 `
  --microsoft-voice en-US-MichelleNeural
```

### 5. Add Web UI support

In `src/tts_from_youtube/ui.py`:

- Add `Webpage` to the Source dropdown.
- Rename the URL textbox to `URL`, and explain that it accepts YouTube when Source is YouTube and an article URL when Source is Webpage.
- Add a webpage timeout control in a small Webpage settings accordion, default 30 seconds, reasonable range 5-120.
- In `_run`, when source is Webpage:
  - reject Download-only;
  - require the URL textbox;
  - call `run_webpage` once;
  - collect artifacts using the existing collector.
- Do not fall through to `run_many`, because that invokes `yt-dlp`.
- Update source coercion so valid `Webpage` values survive. Preserve the current defensive handling for malformed Gradio dropdown payloads.
- Update UI notes to explain review-first use and network/privacy behavior.

### 6. Tests (required before documentation is considered complete)

Do not make tests access the internet. Mock the response/fetch boundary.

Create `tests/test_webpage.py` covering:

1. A representative article HTML fixture extracts title and main paragraphs but excludes nav, footer, scripts, and comments.
2. Relative/redirected final URL is recorded correctly (use a full mocked final URL).
3. Non-HTTP schemes (`file:`, `ftp:`) are rejected.
4. Credential-bearing URLs are rejected.
5. A clearly non-HTML content type is rejected.
6. More than 10 MiB is rejected without an unbounded read.
7. Empty/very short or navigation-only extraction gives the actionable fallback error.
8. Network and HTTP failures become a stable user-facing extraction error.
9. `run_webpage` writes raw, clean, and manifest artifacts and does not invoke ASR.
10. With mocked polishing, `run_webpage` writes `.transcript_tts.txt` and synthesizes that returned text, not the pre-polish text.
11. Authored words such as `like` and `I mean` remain in webpage text after basic cleanup.
12. Manifest contains `input_kind=webpage`, original URL, final URL, polish metadata, TTS metadata, and artifact paths.

Extend UI/CLI smoke coverage if such tests exist. At minimum, verify `y2tts webpage --help` exits successfully.

Use dependency injection or monkeypatch a small `_open_url` helper rather than patching many internals. If Trafilatura output varies between versions, test semantic inclusion/exclusion instead of exact whitespace for extractor unit tests.

## Documentation changes

Update `README.md` in all relevant places:

- title/description: YouTube, webpage, or local input;
- Features / Inputs: ordinary webpage articles;
- pipeline explanation: webpage skips ASR;
- quickstart examples above;
- output artifacts and manifest source fields;
- limitations: paywalls, login pages, bot protection, JavaScript-only rendering, and unusually structured pages may fail; upload saved text as fallback;
- privacy: fetching reveals the URL and IP address to the site; remote polishing and Microsoft TTS transmit extracted text; Ollama + Piper keeps post-fetch processing local.

Do not claim support for authenticated browser sessions or JavaScript rendering in this patch.

## Suggested implementation order

Execute these steps in order and keep the test suite passing after each major step:

1. Add dependency and `webpage.py`.
2. Add extractor unit tests and make them pass.
3. Refactor `_run_text`; confirm all existing tests still pass.
4. Add `run_webpage` and pipeline tests.
5. Add the CLI command and verify help.
6. Add Web UI routing and labels.
7. Update README.
8. Run all verification commands.

## Verification commands

From the repository root in PowerShell:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\ruff.exe check --no-cache src tests
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m typer src.tts_from_youtube.cli webpage --help
git -c safe.directory=V:/home/j/code/github/tts_from_youtube_v2 diff --check
```

If the editable environment does not yet contain Trafilatura, install the updated project dependencies before running tests. Do not weaken or skip tests because the dependency is missing.

## Acceptance checklist

- [x] Existing YouTube, playlist, audio/video, TXT, Markdown, VTT, and PDF behavior still passes tests.
- [x] A normal article URL produces raw and cleaned text artifacts without ASR.
- [x] Optional polish uses the current validation/checkpoint system.
- [x] TTS consumes polished text when polish is enabled and cleaned text otherwise.
- [x] CLI and Web UI both expose webpage input explicitly.
- [x] Webpage failures are bounded, readable, and suggest a local TXT fallback.
- [x] Network, privacy, and unsupported-page limitations are documented accurately.
- [x] Ruff, pytest, CLI help, and `diff --check` pass.

## Post-implementation notes

- Reddit post URLs are handled through Reddit's thread response instead of generic article extraction, so the output includes the original post plus returned nested comments and replies.
- Logged-in pages can use browser cookies or a Netscape-format `cookies.txt` file.
- On Windows, Brave/Chrome direct cookie reads may fail with app-bound cookie encryption. Use Firefox cookies or an exported `cookies.txt` file in that case.
- Firefox does not normally need to stay open for cookie use; staying logged in is what matters. If cookie loading reports a lock or permission error, close Firefox completely and retry.

## Further improvements after this patch

These are worthwhile but out of scope for the first webpage implementation:

1. Add an explicit reviewed-text second-pass command so synthesis can reuse an already reviewed `.transcript_tts.txt` without refetching or repolishing.
2. Require explicit consent for Microsoft TTS in the UI/CLI, similar to remote polishing, because transcript text is sent online.
3. Replace the duplicated `RunConfig` option lists in `run`, `local`, and `webpage` with a carefully tested configuration builder.
4. Convert Markdown syntax to listenable plain text before TTS rather than relying only on the polish prompt.
5. Add cancellation checks between webpage fetch, polish chunks, and TTS chunks.
6. Make filler removal opt-in for all authored documents and configurable for ASR transcripts.
7. Add an optional browser-rendered extractor later, with explicit authentication/privacy controls, for JavaScript-only pages.
