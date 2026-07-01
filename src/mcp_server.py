#!/usr/bin/env python3
"""
MCP server for EduAgent-OS.

Exposes the lecture-to-study-materials pipeline (media ingestion, local
transcription, parallel note/flashcard synthesis, and structured hallucination
evaluation) as MCP tools. Any MCP-compatible client (Claude Desktop, Claude
Code, or another agent) can call these tools to drive EduAgent-OS as a
building block rather than a standalone script.

Run directly (stdio transport):
    python -m src.mcp_server

Requires GEMINI_API_KEY to be set in the environment before process_lecture
is called. See README.md for Claude Desktop / Claude Code configuration.
"""

import json
import os

from pydantic import BaseModel, ConfigDict, Field
from mcp.server.fastmcp import FastMCP

from src.config import (
    WORKSPACE_DIR,
    NOTES_PATH,
    FLASHCARDS_PATH,
    EVALUATION_PATH,
    TRANSCRIPT_PATH,
)
from src.main import run_educational_pipeline

mcp = FastMCP("eduagent_mcp")


class ProcessLectureInput(BaseModel):
    """Input model for running the full pipeline against a lecture source."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    input_source: str = Field(
        ...,
        description=(
            "A YouTube URL (e.g. 'https://www.youtube.com/watch?v=VIDEO_ID') or an "
            "absolute path to a local audio file reachable from the workspace "
            "directory (e.g. '/app/workspace/lecture.mp3')."
        ),
        min_length=1,
        max_length=2048,
    )


def _read_file(path: str, label: str) -> str:
    """Shared file-read helper with a consistent, actionable error message."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{label} not found at '{path}'. Call 'process_lecture' first to "
            "generate it, then retry this tool."
        )
    with open(path, "r") as f:
        return f.read()


def _handle_error(e: Exception) -> str:
    """Consistent error formatting across all tools."""
    if isinstance(e, FileNotFoundError):
        return f"Error: {e}"
    if "GEMINI_API_KEY" in str(e) or isinstance(e, PermissionError):
        return (
            "Error: Missing or invalid GEMINI_API_KEY. Set it in the environment "
            "(or .env file) before calling process_lecture."
        )
    return f"Error: {type(e).__name__}: {e}"


@mcp.tool(
    name="process_lecture",
    annotations={
        "title": "Process Lecture Into Study Materials",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def process_lecture(params: ProcessLectureInput) -> str:
    """Run the full EduAgent-OS pipeline against a YouTube lecture or audio file.

    This tool ingests the source (downloading and extracting audio via yt-dlp
    for YouTube URLs, or reading directly for local files), transcribes it
    locally with faster-whisper, dispatches two parallel Gemini agents (an
    Academic Synthesis Specialist producing hierarchical Markdown study notes,
    and an Educational Taxonomist producing an Anki-compatible Q&A flashcard
    table), and finally runs a structured verification pass that audits both
    outputs against the transcript for factual consistency and hallucination
    risk. All artifacts are written to the shared workspace directory and the
    evaluation is also returned inline in this response.

    This is the expensive, long-running tool in this server (transcription and
    three sequential/parallel LLM calls) - prefer get_notes / get_flashcards /
    get_evaluation / get_transcript for re-reading results from a prior run.

    Args:
        params (ProcessLectureInput): Validated input containing:
            - input_source (str): YouTube URL or local audio file path to process

    Returns:
        str: JSON-formatted string with the schema:
        {
            "status": "success",
            "workspace_dir": str,      # absolute path artifacts were written to
            "notes_path": str,
            "flashcards_path": str,
            "evaluation_path": str,
            "evaluation": {
                "factual_consistency_score": float,   # 0.0-1.0
                "summary_quality_score": float,       # 0.0-1.0
                "hallucination_detected": bool,
                "missing_critical_terms": [str, ...],
                "key_concepts_covered": [str, ...]
            }
        }

        On failure: "Error: <message>" describing what went wrong (e.g. missing
        GEMINI_API_KEY, unreachable URL, or an unresolvable local file path).

    Examples:
        - Use when: "Turn this lecture into study notes: https://youtube.com/watch?v=..."
        - Use when: "Process the audio file at /app/workspace/lecture.mp3"
        - Don't use when: You already ran this and just want the notes or
          flashcards back (use get_notes / get_flashcards instead - they are
          near-instant disk reads instead of a full pipeline run)
    """
    try:
        await run_educational_pipeline(params.input_source)
    except Exception as e:
        return _handle_error(e)

    try:
        evaluation_raw = _read_file(EVALUATION_PATH, "Evaluation report")
        evaluation_parsed = json.loads(evaluation_raw)
    except Exception as e:
        return _handle_error(e)

    response = {
        "status": "success",
        "workspace_dir": WORKSPACE_DIR,
        "notes_path": NOTES_PATH,
        "flashcards_path": FLASHCARDS_PATH,
        "evaluation_path": EVALUATION_PATH,
        "evaluation": evaluation_parsed,
    }
    return json.dumps(response, indent=2)


@mcp.tool(
    name="get_notes",
    annotations={
        "title": "Get Generated Study Notes",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def get_notes() -> str:
    """Retrieve the Markdown study notes generated by the most recent run.

    Reads 'notes.md' from the workspace directory. This does not trigger any
    processing - call 'process_lecture' first if no run has completed yet.

    Returns:
        str: The full Markdown contents of the generated study notes
        (structured with '# Summary' and '## Methodology' style headers), or
        "Error: <message>" if no notes file exists yet.
    """
    try:
        return _read_file(NOTES_PATH, "Study notes")
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="get_flashcards",
    annotations={
        "title": "Get Generated Flashcards",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def get_flashcards() -> str:
    """Retrieve the Anki-compatible Q&A flashcard table from the most recent run.

    Reads 'flashcards.md' from the workspace directory. This does not trigger
    any processing - call 'process_lecture' first if no run has completed yet.

    Returns:
        str: The full Markdown Q&A table of flashcards, or "Error: <message>"
        if no flashcards file exists yet.
    """
    try:
        return _read_file(FLASHCARDS_PATH, "Flashcards")
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="get_evaluation",
    annotations={
        "title": "Get Hallucination and Quality Evaluation Report",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def get_evaluation() -> str:
    """Retrieve the structured quality/hallucination audit from the most recent run.

    Reads 'evaluation.json' from the workspace directory and returns it parsed
    and re-serialized. This does not trigger any processing - call
    'process_lecture' first if no run has completed yet.

    Returns:
        str: JSON-formatted string matching the LectureEvaluation schema:
        {
            "factual_consistency_score": float,   # 0.0-1.0
            "summary_quality_score": float,       # 0.0-1.0
            "hallucination_detected": bool,
            "missing_critical_terms": [str, ...],
            "key_concepts_covered": [str, ...]
        }
        Or "Error: <message>" if no evaluation file exists yet.
    """
    try:
        raw = _read_file(EVALUATION_PATH, "Evaluation report")
    except Exception as e:
        return _handle_error(e)
    try:
        return json.dumps(json.loads(raw), indent=2)
    except json.JSONDecodeError:
        return raw


@mcp.tool(
    name="get_transcript",
    annotations={
        "title": "Get Raw Lecture Transcript",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def get_transcript() -> str:
    """Retrieve the raw timestamped transcript from the most recent pipeline run.

    Reads 'transcript.txt' from the workspace directory. Useful for spot-checking
    the evaluation report's claims (e.g. missing_critical_terms) against the
    original source material without re-running transcription.

    Returns:
        str: The full timestamped transcript text (one "[start s - end s] text"
        line per dialogue segment), or "Error: <message>" if no transcript file
        exists yet.
    """
    try:
        return _read_file(TRANSCRIPT_PATH, "Transcript")
    except Exception as e:
        return _handle_error(e)


if __name__ == "__main__":
    mcp.run()
