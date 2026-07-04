import os
import re
from urllib.parse import urlparse

from yt_dlp import YoutubeDL

from src.config import WORKSPACE_DIR

# Only YouTube hosts are allowed for remote ingestion. Anything else is
# rejected before any network call, which prevents the public demo from being
# used as an SSRF proxy against arbitrary or internal endpoints.
ALLOWED_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
}


def extract_video_id(url: str):
    """Extracts the unique video ID from a standard YouTube URL string.

    Returns None when no 11-character video ID can be found, so callers can
    reject the URL instead of silently sharing a fallback cache filename
    between unrelated inputs.
    """
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11})(?:[?&\/]|$)'
    match = re.search(pattern, url)
    return match.group(1) if match else None


def _resolve_local_path(input_source: str, workspace_dir: str) -> str:
    """Resolve a local file path, refusing anything outside the workspace.

    Without this check, any visitor to the hosted demo (or any MCP client)
    could point the pipeline at arbitrary readable files inside the container.
    """
    real = os.path.realpath(input_source)
    ws_real = os.path.realpath(workspace_dir)
    if not os.path.isfile(real):
        raise FileNotFoundError(
            f"Input source target '{input_source}' could not be resolved."
        )
    if not (real == ws_real or real.startswith(ws_real + os.sep)):
        raise PermissionError(
            f"Local file access is restricted to the workspace directory "
            f"('{ws_real}'). Move the file there and retry."
        )
    return real


def process_input_source(input_source: str, workspace_dir: str = None) -> str:
    """Ingests a source path. Checks for a cached file match before spinning up network fetches."""
    workspace_dir = workspace_dir or WORKSPACE_DIR

    # Handle YouTube URL strings
    if input_source.startswith("http://") or input_source.startswith("https://"):
        host = (urlparse(input_source).hostname or "").lower()
        if host not in ALLOWED_HOSTS:
            raise ValueError(
                f"Refusing to fetch from host '{host}'. Only YouTube URLs are "
                f"supported ({', '.join(sorted(ALLOWED_HOSTS))})."
            )

        video_id = extract_video_id(input_source)
        if not video_id:
            raise ValueError(
                f"Could not extract a video ID from '{input_source}'. "
                "Provide a standard YouTube watch or share URL."
            )

        cached_file_path = os.path.join(workspace_dir, f"{video_id}.mp3")

        # Performance Cache Check
        if os.path.exists(cached_file_path):
            print(f"[Fetcher] Cache hit -- skipping download: {video_id}.mp3")
            return cached_file_path

        print(f"[Fetcher] Cache miss. Ingesting remote URL stream: {input_source}")

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(workspace_dir, f"{video_id}.%(ext)s"),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
        }

        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([input_source])

        print(f"[Fetcher] Stream downloaded successfully: {video_id}.mp3")
        return cached_file_path

    # Handle direct local file paths (sandboxed to the workspace directory)
    resolved = _resolve_local_path(input_source, workspace_dir)
    print(f"[Fetcher] Local file path recognized: {resolved}")
    return resolved
