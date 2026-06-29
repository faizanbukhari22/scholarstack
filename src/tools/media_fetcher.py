import os
from pathlib import Path
import yt_dlp

def process_input_source(source_path: str, workspace_dir: str = "/app/workspace") -> str:
    local_path = Path(source_path)
    if local_path.is_file() or os.path.exists(Path(workspace_dir) / source_path):
        resolved_path = local_path if local_path.is_absolute() else Path(workspace_dir) / source_path
        print(f"[Fetcher] Target found locally: {resolved_path.name}")
        return str(resolved_path.resolve())

    print(f"[Fetcher] Ingesting remote URL stream: {source_path}")
    out_path = Path(workspace_dir)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',
        }],
        'outtmpl': str(out_path / '%(id)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(source_path, download=True)
            target_file = out_path / f"{info['id']}.mp3"
            print(f"[Fetcher] Stream downloaded successfully: {target_file.name}")
            return str(target_file.resolve())
    except Exception as e:
        print(f"[Fetcher Error] Failed to extract from URL source: {e}")
        raise e
