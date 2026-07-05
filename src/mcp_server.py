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
import glob

from pydantic import BaseModel, ConfigDict, Field
from mcp.server.fastmcp import FastMCP

from src.config import (
    LIBRARY_DIR,
    get_env,
    get_lecture_paths,
)
from src.tools.media_fetcher import get_or_create_lecture_dir
from src.main import run_educational_pipeline
from src.tools.pdf_generator import compile_markdown_to_pdf

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
    force: bool = Field(
        False,
        description="Force regeneration of study materials (notes, flashcards, evaluation) from cached transcript without re-downloading or re-transcribing."
    )
    force_all: bool = Field(
        False,
        description="Wipe existing cache completely and force a full pipeline re-run (redownload, retranscribe, regenerate)."
    )


def resolve_lecture_paths_for_mcp(lecture_id: str = None) -> dict:
    """Resolves paths for a specified lecture_id or the most recently modified complete lecture."""
    if lecture_id:
        # Check folders ending in __lecture_id
        search_pattern = os.path.join(LIBRARY_DIR, f"*__{lecture_id}")
        matches = glob.glob(search_pattern)
        if not matches:
            # Check exact folder name match
            exact_path = os.path.join(LIBRARY_DIR, lecture_id)
            if os.path.isdir(exact_path):
                matches = [exact_path]
        if not matches:
            raise FileNotFoundError(f"Lecture with ID/folder '{lecture_id}' not found in library.")
        return get_lecture_paths(matches[0])

    # Fallback to the latest completed run
    folders = glob.glob(os.path.join(LIBRARY_DIR, "*__*"))
    completed_folders = []
    for folder in folders:
        meta_path = os.path.join(folder, "meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("status") == "complete":
                    mtime = os.path.getmtime(meta_path)
                    completed_folders.append((mtime, folder))
            except Exception:
                pass
    if not completed_folders:
        raise FileNotFoundError("No completed lectures found in library. Call 'process_lecture' first.")

    # Sort by mtime descending (newest first)
    completed_folders.sort(key=lambda x: x[0], reverse=True)
    return get_lecture_paths(completed_folders[0][1])


def _read_file(path: str, label: str) -> str:
    """Shared file-read helper with a consistent, actionable error message."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{label} not found at '{path}'. Call 'process_lecture' first to "
            "generate it, then retry this tool."
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _handle_error(e: Exception) -> str:
    """Consistent error formatting across all tools, with API key redaction."""
    msg = str(e)
    # Redact API key fragments from error messages
    api_key = get_env("GEMINI_API_KEY")
    if api_key and api_key in msg:
        msg = msg.replace(api_key, "[REDACTED]")
    if api_key and len(api_key) > 8:
        msg = msg.replace(api_key[:8], "[REDACTED]")

    if isinstance(e, FileNotFoundError):
        return f"Error: {msg}"
    if "GEMINI_API_KEY" in msg or isinstance(e, PermissionError):
        return (
            "Error: Missing or invalid GEMINI_API_KEY. Set it in the environment "
            "(or .env file) before calling process_lecture."
        )
    return f"Error: {type(e).__name__}: {msg}"


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

    This tool ingests the source, transcribes it locally, dispatches Gemini agents
    to generate study notes and flashcards, and performs a hallucination verification check.
    All outputs are saved in the persistent lecture-specific directory under the library folder.

    Args:
        params (ProcessLectureInput): Validated input containing:
            - input_source (str): YouTube URL or local audio file path to process
            - force (bool): Regenerate Gemini study materials only
            - force_all (bool): Retranscribe and redownload everything
    """
    try:
        await run_educational_pipeline(
            params.input_source,
            force=params.force,
            force_all=params.force_all
        )
    except Exception as e:
        return _handle_error(e)

    try:
        lecture_dir, _ = get_or_create_lecture_dir(params.input_source, LIBRARY_DIR)
        paths = get_lecture_paths(lecture_dir)
        evaluation_raw = _read_file(paths["evaluation"], "Evaluation report")
        evaluation_parsed = json.loads(evaluation_raw)
    except Exception as e:
        return _handle_error(e)

    response = {
        "status": "success",
        "workspace_dir": paths["workspace"],
        "notes_path": paths["notes"],
        "flashcards_path": paths["flashcards"],
        "evaluation_path": paths["evaluation"],
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
async def get_notes(lecture_id: str = None) -> str:
    """Retrieve the Markdown study notes generated for a specific lecture ID or the latest run.

    Args:
        lecture_id (str, optional): The YouTube video ID or local content hash to retrieve notes for.
                                    If not specified, returns the most recently completed lecture.
    """
    try:
        paths = resolve_lecture_paths_for_mcp(lecture_id)
        return _read_file(paths["notes"], "Study notes")
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
async def get_flashcards(lecture_id: str = None) -> str:
    """Retrieve the Anki-compatible Q&A flashcard table for a specific lecture ID or the latest run.

    Args:
        lecture_id (str, optional): The YouTube video ID or local content hash.
                                    If not specified, returns the most recently completed lecture.
    """
    try:
        paths = resolve_lecture_paths_for_mcp(lecture_id)
        return _read_file(paths["flashcards"], "Flashcards")
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
async def get_evaluation(lecture_id: str = None) -> str:
    """Retrieve the structured quality/hallucination audit for a specific lecture ID or the latest run.

    Args:
        lecture_id (str, optional): The YouTube video ID or local content hash.
                                    If not specified, returns the most recently completed lecture.
    """
    try:
        paths = resolve_lecture_paths_for_mcp(lecture_id)
        raw = _read_file(paths["evaluation"], "Evaluation report")
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
async def get_transcript(lecture_id: str = None) -> str:
    """Retrieve the raw timestamped transcript for a specific lecture ID or the latest run.

    Args:
        lecture_id (str, optional): The YouTube video ID or local content hash.
                                    If not specified, returns the most recently completed lecture.
    """
    try:
        paths = resolve_lecture_paths_for_mcp(lecture_id)
        return _read_file(paths["transcript"], "Transcript")
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="export_pdf",
    annotations={
        "title": "Export Study Guide to PDF",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def export_pdf(lecture_id: str = None) -> str:
    """Compile Markdown study notes into a print-ready PDF document for a specific lecture or the latest run.

    Args:
        lecture_id (str, optional): The YouTube video ID or local content hash.
                                    If not specified, targets the most recently completed lecture.
    """
    try:
        paths = resolve_lecture_paths_for_mcp(lecture_id)
        if not os.path.exists(paths["notes"]):
            raise FileNotFoundError("Study notes file (notes.md) not found.")
        
        compile_markdown_to_pdf(paths["notes"], paths["pdf"])
        
        response = {
            "status": "success",
            "pdf_path": paths["pdf"],
            "message": f"Successfully compiled study guide PDF at: {paths['pdf']}"
        }
        return json.dumps(response, indent=2)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="list_library",
    annotations={
        "title": "List Processed Lecture Library",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def list_library() -> str:
    """List all processed lectures currently stored in the automated library.

    Returns:
        str: JSON-formatted list of completed lectures in the library, sorted by creation date descending.
    """
    try:
        folders = glob.glob(os.path.join(LIBRARY_DIR, "*__*"))
        library_items = []
        for folder in folders:
            meta_path = os.path.join(folder, "meta.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if meta.get("status") == "complete":
                        library_items.append({
                            "id": meta.get("id"),
                            "title": meta.get("title"),
                            "source": meta.get("source"),
                            "created_at": meta.get("created_at"),
                            "folder_name": os.path.basename(folder)
                        })
                except Exception:
                    pass
        # Sort by created_at descending (newest first)
        library_items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return json.dumps(library_items, indent=2)
    except Exception as e:
        return _handle_error(e)


if __name__ == "__main__":
    mcp.run()
