FROM --platform=linux/arm64 python:3.11-slim

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

# Create an isolated workspace directory inside the container environment
RUN mkdir -p /app/workspace

# Copy code modules over to the container
COPY src/ /app/src/

ENV PYTHONPATH=/app

CMD ["python", "src/main.py"]
