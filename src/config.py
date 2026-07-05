"""
Shared runtime configuration for EduAgent-OS.

Centralizes the workspace directory resolution so the same pipeline code
works identically whether it is invoked as the Docker Compose batch job
(src/main.py) or as a local process spawned by an MCP client (src/mcp_server.py).
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Module-level store for .env values. Avoids polluting os.environ (which is
# inherited by every subprocess — ffmpeg, yt-dlp — and can leak API keys if
# those processes log their environment on crash).
_env_store: dict[str, str] = {}


def _load_env_file():
    """Load keys from .env if present into the module-level store."""
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip()
                        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                            val = val[1:-1]
                        if key:
                            _env_store[key] = val
        except Exception as e:
            print(f"[Config] Warning: Failed to load .env file: {e}")


def get_env(key: str, default: str = "") -> str:
    """Read a config value: os.environ takes priority, then .env store."""
    return os.getenv(key) or _env_store.get(key, default)


_load_env_file()


def _resolve_workspace_dir() -> str:
    """Resolve the workspace directory across container and local execution.

    Resolution order:
    1. Explicit WORKSPACE_DIR environment variable override.
    2. The container-baked path (/app/workspace) when running inside Docker.
    3. A local <project_root>/workspace directory for host execution
       (e.g. when an MCP client spawns this code directly on the user's machine).
    """
    override = os.getenv("WORKSPACE_DIR")
    if override:
        return override

    if os.path.isdir("/app/workspace"):
        return "/app/workspace"

    return os.path.join(PROJECT_ROOT, "workspace")


WORKSPACE_DIR = _resolve_workspace_dir()
os.makedirs(WORKSPACE_DIR, exist_ok=True)


LIBRARY_DIR = os.path.join(WORKSPACE_DIR, "library")
os.makedirs(LIBRARY_DIR, exist_ok=True)


def get_lecture_paths(lecture_dir: str):
    """Return paths for all artifacts inside a specific lecture's folder."""
    os.makedirs(lecture_dir, exist_ok=True)
    return {
        "workspace": lecture_dir,
        "audio": os.path.join(lecture_dir, "audio.mp3"),
        "transcript": os.path.join(lecture_dir, "transcript.txt"),
        "notes": os.path.join(lecture_dir, "notes.md"),
        "flashcards": os.path.join(lecture_dir, "flashcards.md"),
        "evaluation": os.path.join(lecture_dir, "evaluation.json"),
        "pdf": os.path.join(lecture_dir, "notes.pdf"),
        "meta": os.path.join(lecture_dir, "meta.json"),
        "lock": os.path.join(lecture_dir, ".lock"),
    }
