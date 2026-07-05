import os
import re
import glob
import json
import hashlib
import subprocess
import unicodedata
from datetime import datetime, timezone
from urllib.parse import urlparse

from yt_dlp import YoutubeDL

from src.config import WORKSPACE_DIR, get_lecture_paths

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


def slugify(text: str) -> str:
    """Normalize and slugify a string to make it safe for file paths."""
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    text = text.strip('-')
    return text[:60] or "lecture"


def compute_file_hash(file_path: str) -> str:
    """Compute a SHA-256 hash of the local file in chunks for identity tracking."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


def extract_audio_from_local(input_path: str, output_path: str):
    """Extract audio from local video/audio file to mp3 using ffmpeg."""
    print(f"[Fetcher] Extracting audio from local file '{input_path}' to '{output_path}'...")
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn",
        "-acodec", "libmp3lame",
        "-ar", "44100",
        "-ab", "192k",
        output_path
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"[Fetcher] Audio extracted successfully: {output_path}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg audio extraction failed: {e}")


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


def write_meta_atomic(meta_path: str, meta: dict) -> None:
    """Write meta.json via a temp file + atomic rename.

    meta.json is the trust anchor for cache hits (status == "complete"), so a
    crash mid-write must never leave truncated JSON behind.
    """
    tmp_path = meta_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp_path, meta_path)


def get_or_create_lecture_dir(input_source: str, library_dir: str) -> tuple[str, dict]:
    """Finds or creates a lecture directory for the given input source.

    Returns:
        tuple[str, dict]: The path to the lecture folder and the metadata dictionary.
    """
    os.makedirs(library_dir, exist_ok=True)
    is_url = input_source.startswith("http://") or input_source.startswith("https://")

    if is_url:
        host = (urlparse(input_source).hostname or "").lower()
        if host not in ALLOWED_HOSTS:
            raise ValueError(
                f"Refusing to fetch from host '{host}'. Only YouTube URLs are "
                f"supported ({', '.join(sorted(ALLOWED_HOSTS))})."
            )
        video_id = extract_video_id(input_source)
        if not video_id:
            raise ValueError(f"Could not extract a video ID from '{input_source}'.")
        identifier = f"yt_{video_id}"
    else:
        resolved = _resolve_local_path(input_source, WORKSPACE_DIR)
        file_hash = compute_file_hash(resolved)
        identifier = f"local_{file_hash}"

    # Search for folder ending in __<identifier>
    search_pattern = os.path.join(library_dir, f"*___*") # wait, search_pattern in local file is *__identifier
    search_pattern = os.path.join(library_dir, f"*__{identifier}")
    matches = glob.glob(search_pattern)

    if matches:
        lecture_dir = matches[0]
        meta_path = os.path.join(lecture_dir, "meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                return lecture_dir, meta
            except Exception as e:
                print(f"[Fetcher] Warning: Failed to read meta.json: {e}")

        dir_name = os.path.basename(lecture_dir)
        title_part = dir_name.split("__")[0]
        meta = {
            "id": identifier,
            "title": title_part.replace("-", " ").title(),
            "source": input_source,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "processing",
            "pipeline_version": "1.0",
        }
        # Persist the reconstructed metadata so the folder heals itself and
        # subsequent runs do not have to rebuild it from the folder name.
        write_meta_atomic(meta_path, meta)
        return lecture_dir, meta

    # Cache miss - fetch title and create folder
    title = "lecture"
    if is_url:
        print(f"[Fetcher] Fetching video metadata for: {input_source}")
        ydl_opts = {'quiet': True, 'no_warnings': True}
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(input_source, download=False)
                title = info.get('title', 'Unknown YouTube Video')
        except Exception as e:
            print(f"[Fetcher] Warning: Failed to fetch YouTube title: {e}")
            title = f"youtube-video-{video_id}"
    else:
        resolved = _resolve_local_path(input_source, WORKSPACE_DIR)
        title = os.path.splitext(os.path.basename(resolved))[0]

    slug = slugify(title)
    folder_name = f"{slug}__{identifier}"
    lecture_dir = os.path.join(library_dir, folder_name)
    os.makedirs(lecture_dir, exist_ok=True)

    meta = {
        "id": identifier,
        "title": title,
        "source": input_source,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "processing",
        "pipeline_version": "1.0",
    }

    write_meta_atomic(os.path.join(lecture_dir, "meta.json"), meta)

    return lecture_dir, meta


def process_input_source(input_source: str, lecture_dir: str) -> str:
    """Ingests a source path and extracts audio to audio.mp3 under the lecture folder."""
    paths = get_lecture_paths(lecture_dir)
    audio_path = paths["audio"]

    if os.path.exists(audio_path):
        print(f"[Fetcher] Audio cache hit: {audio_path}")
        return audio_path

    is_url = input_source.startswith("http://") or input_source.startswith("https://")

    if is_url:
        print(f"[Fetcher] Downloading remote URL stream: {input_source}")
        temp_out = os.path.join(lecture_dir, "temp_download.%(ext)s")
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': temp_out,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
        }

        try:
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([input_source])

            # H3: Find whatever yt-dlp actually produced (may not be .mp3 if
            # ffmpeg is missing or the postprocessor chose a different codec).
            temp_files = glob.glob(os.path.join(lecture_dir, "temp_download.*"))
            if not temp_files:
                raise RuntimeError(
                    "yt-dlp download produced no output file. Ensure ffmpeg is "
                    "installed and the URL is a valid YouTube video."
                )
            # Prefer the postprocessed .mp3 if multiple temp files exist
            # (e.g. the original container alongside the extracted audio);
            # glob order is filesystem-dependent and must not decide this.
            temp_files.sort(key=lambda p: (not p.endswith(".mp3"), p))
            os.replace(temp_files[0], audio_path)
        finally:
            # Clean up any leftover temp files (including partial downloads
            # from OOM kills or Docker stops).
            for leftover in glob.glob(os.path.join(lecture_dir, "temp_download.*")):
                try:
                    os.remove(leftover)
                except OSError:
                    pass

        print(f"[Fetcher] Stream downloaded successfully: {audio_path}")
        return audio_path

    # Local file
    resolved = _resolve_local_path(input_source, WORKSPACE_DIR)
    extract_audio_from_local(resolved, audio_path)
    return audio_path
