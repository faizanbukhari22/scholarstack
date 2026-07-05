import os
from faster_whisper import WhisperModel

# H4: Cache the Whisper model at module level so it is loaded once and reused
# across all requests. Each model load reads ~150MB from disk — under
# demo.queue(max_size=5) that means 5 concurrent loads without this cache.
_model = None


def _get_model() -> WhisperModel:
    """Return the cached WhisperModel, loading it on first call."""
    global _model
    if _model is not None:
        return _model

    baked_root = os.getenv("WHISPER_MODEL_PATH", "/app/models")

    if os.path.exists(baked_root):
        print(f"[Transcriber] Attempting to load Whisper model (base) from baked cache: {baked_root}...")
        try:
            _model = WhisperModel(
                model_size_or_path="base",
                device="cpu",
                compute_type="int8",
                download_root=baked_root
            )
            return _model
        except Exception as e:
            print(f"[Transcriber] Warning: Failed to load model from baked cache ({type(e).__name__}: {e}).")
            print("[Transcriber] Falling back to standard Hugging Face Hub download...")

    try:
        _model = WhisperModel(
            model_size_or_path="base",
            device="cpu",
            compute_type="int8"
        )
        return _model
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize Whisper model from both baked cache and Hugging Face Hub: {e}"
        ) from e


def transcribe_audio_file(audio_path: str) -> str:
    """Loads a pre-baked model from the local image footprint and runs transcription."""
    model = _get_model()

    # beam_size=1 (greedy) keeps memory flat; higher values keep N hypotheses
    # in memory and can OOM-kill the container (exit 137) on long lectures.
    beam_size = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
    print(f"[Transcriber] Processing audio timeline for: {os.path.basename(audio_path)} (beam_size={beam_size})")
    segments, info = model.transcribe(audio_path, beam_size=beam_size)

    transcript_segments = []
    for segment in segments:
        transcript_segments.append(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")

    return "\n".join(transcript_segments)
