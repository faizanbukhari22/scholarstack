import os
from faster_whisper import WhisperModel

def transcribe_audio_file(audio_path: str) -> str:
    """Loads a pre-baked model from the local image footprint and runs transcription."""
    baked_root = os.getenv("WHISPER_MODEL_PATH", "/app/models")
    model = None

    if os.path.exists(baked_root):
        print(f"[Transcriber] Attempting to load Whisper model (base) from baked cache: {baked_root}...")
        try:
            model = WhisperModel(
                model_size_or_path="base",
                device="cpu",
                compute_type="int8",
                download_root=baked_root
            )
        except Exception as e:
            print(f"[Transcriber] Warning: Failed to load model from baked cache ({type(e).__name__}: {e}).")
            print("[Transcriber] Falling back to standard Hugging Face Hub download...")

    if model is None:
        try:
            model = WhisperModel(
                model_size_or_path="base",
                device="cpu",
                compute_type="int8"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Whisper model from both baked cache and Hugging Face Hub: {e}"
            ) from e
    
    # beam_size=1 (greedy) keeps memory flat; higher values keep N hypotheses
    # in memory and can OOM-kill the container (exit 137) on long lectures.
    beam_size = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
    print(f"[Transcriber] Processing audio timeline for: {os.path.basename(audio_path)} (beam_size={beam_size})")
    segments, info = model.transcribe(audio_path, beam_size=beam_size)
    
    transcript_segments = []
    for segment in segments:
        transcript_segments.append(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")
        
    return "\n".join(transcript_segments)
