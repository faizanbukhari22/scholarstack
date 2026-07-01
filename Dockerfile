ARG BUILDPLATFORM
FROM --platform=${BUILDPLATFORM:-linux/amd64} python:3.11-slim

# Install system dependencies (ffmpeg is required by both yt-dlp and faster-whisper)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency mappings and install packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the Whisper base model weights into the image at build time.
# download_root='/app/models' writes to an explicit, controlled path inside
# the image layer. At runtime, transcriber.py reads WHISPER_MODEL_PATH and
# passes it as download_root so faster-whisper loads from disk instantly,
# with zero network activity and zero validation delay.
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8', download_root='/app/models')"

# Create an isolated workspace directory inside the container environment
RUN mkdir -p /app/workspace

# Copy code modules over to the container
COPY src/ /app/src/
COPY app.py /app/app.py

ENV PYTHONPATH=/app
# Explicit path to baked model weights — read by transcriber.py at runtime
ENV WHISPER_MODEL_PATH="/app/models"

# Default entrypoint serves the Gradio web demo (used by Hugging Face Spaces,
# which expects a long-running HTTP server on this port). Local batch runs via
# docker-compose override this with `command: python src/main.py`.
EXPOSE 7860
CMD ["python", "app.py"]
