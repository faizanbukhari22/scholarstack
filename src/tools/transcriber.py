import os
from pathlib import Path
from faster_whisper import WhisperModel

def transcribe_audio_file(audio_path: str, model_size: str = "base") -> list:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"[Transcriber Error] Audio target file missing: {audio_path}")

    print(f"[Transcriber] Initializing local Whisper model ({model_size})...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    print(f"[Transcriber] Processing audio timeline for: {Path(audio_path).name}")
    segments, info = model.transcribe(audio_path, beam_size=5)
    
    compiled_segments = []
    for segment in segments:
        compiled_segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })
        
    print(f"[Transcriber] Extraction complete. Captured {len(compiled_segments)} dialogue blocks.")
    return compiled_segments
