from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from yt_dlp import YoutubeDL

from .utils import ensure_dir, sanitize_filename


@dataclass(frozen=True)
class VideoItem:
    url: str
    id: str | None = None
    title: str | None = None


def expand_url(url: str) -> list[VideoItem]:
    """Return a list of VideoItem(s). If url is a playlist, it will be expanded to entries."""
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": "in_playlist",
        "dump_single_json": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    if info is None:
        return [VideoItem(url=url)]

    if "entries" in info and info["entries"]:
        items: list[VideoItem] = []
        for e in info["entries"]:
            if not e:
                continue
            # yt-dlp provides a 'url' field that may be an ID; 'webpage_url' is usually complete
            entry_url = e.get("webpage_url") or e.get("url") or ""
            if entry_url and not entry_url.startswith("http"):
                # fallback: treat as video id
                entry_url = f"https://www.youtube.com/watch?v={entry_url}"
            items.append(VideoItem(url=entry_url, id=e.get("id"), title=e.get("title")))
        return items

    return [VideoItem(url=info.get("webpage_url") or url, id=info.get("id"), title=info.get("title"))]


@dataclass
class DownloadResult:
    audio_path: Path
    title: str
    video_id: str | None
    captions_path: Path | None

@dataclass
class DownloadVideoResult:
    video_path: Path
    title: str
    video_id: str | None


def download_video(
    item: VideoItem,
    out_dir: Path,
    *,
    merge_output_format: str = "mp4",
    fmt: str = "bestvideo+bestaudio/best",
) -> DownloadVideoResult:
    """Download video (or best available) for a single YouTube item.

    - Produces a merged container (default mp4) if separate audio/video streams are used.
    - Requires ffmpeg on PATH (yt-dlp will call it).
    """
    ensure_dir(out_dir)

    tmpl = "%(title)s [%(id)s].%(ext)s"
    ydl_opts = {
        "format": fmt,
        "outtmpl": str(out_dir / tmpl),
        "quiet": True,
        "noplaylist": True,
        "ignoreerrors": True,
        "merge_output_format": merge_output_format,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(item.url, download=True)

    if info is None:
        raise RuntimeError(f"yt-dlp failed to download: {item.url}")

    title = info.get("title") or item.title or "untitled"
    video_id = info.get("id") or item.id

    # Try to locate the merged file. yt-dlp typically reports 'ext' after merge.
    ext = info.get("ext") or merge_output_format
    title_safe = sanitize_filename(title)
    candidates = sorted(out_dir.glob(f"{title_safe}*[{video_id}]*.{ext}")) if video_id else []
    if not candidates:
        # Fallback: newest file with that ext.
        candidates = sorted(out_dir.glob(f"*.{ext}"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        # Last resort: any recent file
        candidates = sorted(out_dir.glob("*.*"), key=lambda p: p.stat().st_mtime, reverse=True)

    return DownloadVideoResult(video_path=candidates[0], title=title, video_id=video_id)


def download_best_audio(
    item: VideoItem,
    out_dir: Path,
    *,
    download_captions: bool = True,
    captions_format: str = "vtt",
    preferred_codec: str = "wav",
) -> DownloadResult:
    """Download best available audio and (optionally) captions."""
    ensure_dir(out_dir)

    codec = preferred_codec.lower()
    if codec not in {"wav", "mp3"}:
        raise ValueError("preferred_codec must be 'wav' or 'mp3'")

    # Use a stable filename template to avoid collisions; include id when available.
    tmpl = "%(title)s [%(id)s].%(ext)s"
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / tmpl),
        "quiet": True,
        "noplaylist": True,
        "ignoreerrors": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": codec,
                "preferredquality": "0",
            }
        ],
    }

    if download_captions:
        ydl_opts.update(
            {
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitlesformat": captions_format,
                "skip_download": False,
            }
        )

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(item.url, download=True)

    if info is None:
        raise RuntimeError(f"yt-dlp failed to download: {item.url}")

    title = info.get("title") or item.title or "untitled"
    video_id = info.get("id") or item.id
    title_safe = sanitize_filename(title)

    # After post-processing, the extension matches requested codec.
    candidates = sorted(out_dir.glob(f"{title_safe}*[{video_id}]*.{codec}")) if video_id else []
    if not candidates:
        candidates = sorted(out_dir.glob(f"*.{codec}"), key=lambda p: p.stat().st_mtime, reverse=True)
    audio_path = candidates[0]

    captions_path = None
    if download_captions:
        # Try to locate matching captions file.
        # yt-dlp naming may append language codes; take newest vtt/srt in folder as fallback.
        cap_candidates = list(out_dir.glob(f"{title_safe}*[{video_id}]*.{captions_format}")) if video_id else []
        if not cap_candidates:
            cap_candidates = list(out_dir.glob(f"*.{captions_format}"))
        if cap_candidates:
            captions_path = sorted(cap_candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    return DownloadResult(audio_path=audio_path, title=title, video_id=video_id, captions_path=captions_path)
