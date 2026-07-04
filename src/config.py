"""
Shared runtime configuration for EduAgent-OS.

Centralizes the workspace directory resolution so the same pipeline code
works identically whether it is invoked as the Docker Compose batch job
(src/main.py) or as a local process spawned by an MCP client (src/mcp_server.py).
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_env_file():
    """Load keys from .env if present and not already in environment."""
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
                        if key and key not in os.environ:
                            os.environ[key] = val
        except Exception as e:
            print(f"[Config] Warning: Failed to load .env file: {e}")


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


def get_workspace_paths(workspace_dir=None):
    """Return the artifact paths for a given workspace directory.

    Passing an explicit directory lets callers (e.g. the Gradio frontend)
    isolate each request in its own workspace instead of sharing the
    module-level default, which prevents cross-user data leakage.
    """
    base = workspace_dir or WORKSPACE_DIR
    os.makedirs(base, exist_ok=True)
    return {
        "workspace": base,
        "transcript": os.path.join(base, "transcript.txt"),
        "notes": os.path.join(base, "notes.md"),
        "flashcards": os.path.join(base, "flashcards.md"),
        "evaluation": os.path.join(base, "evaluation.json"),
        "pdf": os.path.join(base, "notes.pdf"),
    }

TRANSCRIPT_PATH = os.path.join(WORKSPACE_DIR, "transcript.txt")
NOTES_PATH = os.path.join(WORKSPACE_DIR, "notes.md")
FLASHCARDS_PATH = os.path.join(WORKSPACE_DIR, "flashcards.md")
EVALUATION_PATH = os.path.join(WORKSPACE_DIR, "evaluation.json")
PDF_PATH = os.path.join(WORKSPACE_DIR, "notes.pdf")
