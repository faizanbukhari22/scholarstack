import os
import re
from yt_dlp import YoutubeDL
from src.config import WORKSPACE_DIR

def extract_video_id(url: str) -> str:
    """Extracts the unique video ID from a standard YouTube URL string."""
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    return match.group(1) if match else "extracted_audio"

def process_input_source(input_source: str) -> str:
    """Ingests a source path. Checks for a cached file match before spinning up network fetches."""
    workspace_dir = WORKSPACE_DIR
    
    # Handle direct local file paths passed to the container
    if os.path.exists(input_source):
        print(f"[Fetcher] Local file path recognized: {input_source}")
        return input_source

    # Handle YouTube URL strings
    if input_source.startswith("http"):
        video_id = extract_video_id(input_source)
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

    raise FileNotFoundError(f"Input source target '{input_source}' could not be resolved.")
