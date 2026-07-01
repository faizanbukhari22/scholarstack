#!/usr/bin/env python3
"""
Gradio frontend for EduAgent-OS.

Serves the containerized pipeline (ingestion, local transcription, parallel
Gemini synthesis, and structured hallucination verification) as an
interactive web demo. This is the entrypoint used by the Hugging Face
Spaces Docker image (see Dockerfile: CMD ["python", "app.py"]).

Local batch runs via docker-compose are unaffected - docker-compose.yml
overrides the container command back to `python src/main.py`.
"""

import json
import os

import gradio as gr

from src.config import (
    EVALUATION_PATH,
    FLASHCARDS_PATH,
    NOTES_PATH,
)
from src.main import run_educational_pipeline

DESCRIPTION = (
    "Paste a YouTube lecture URL below. EduAgent-OS transcribes it locally with "
    "faster-whisper, dispatches two parallel Gemini agents to write structured "
    "study notes and Anki-style flashcards, then runs a verification pass that "
    "audits both outputs for factual consistency and hallucinations.\n\n"
    "Transcription and generation run on CPU, so processing can take a few "
    "minutes depending on lecture length and the Space's hardware tier."
)


def _read(path: str) -> str:
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    return ""


async def process_lecture(input_source: str, api_key: str = "", progress=gr.Progress(track_tqdm=False)):
    """Run the pipeline and return (status, notes, flashcards, evaluation_json)."""
    empty = ("", "", "")

    if not input_source or not input_source.strip():
        return ("Please paste a YouTube URL or local file path.", *empty)

    # Never mutate os.environ here: on a shared Space that would leak one
    # user's key into every other user's session. Resolve the key locally
    # and pass it explicitly to the pipeline instead.
    effective_key = (api_key or "").strip() or os.getenv("GEMINI_API_KEY", "")
    if not effective_key:
        return (
            "Error: No Gemini API key available. "
            "Provide one in the configuration section, or configure GEMINI_API_KEY on the host.",
            *empty,
        )

    progress(0.05, desc="Initializing pipeline...")
    try:
        def on_progress(fraction: float, desc: str):
            progress(fraction, desc=desc)

        await run_educational_pipeline(
            input_source.strip(),
            progress_callback=on_progress,
            api_key=effective_key,
        )
    except Exception as e:
        return (f"Error: {type(e).__name__}: {e}", *empty)

    progress(0.95, desc="Loading generated artifacts...")

    notes = _read(NOTES_PATH)
    flashcards = _read(FLASHCARDS_PATH)
    evaluation_raw = _read(EVALUATION_PATH)
    try:
        evaluation = json.dumps(json.loads(evaluation_raw), indent=2)
    except (json.JSONDecodeError, TypeError):
        evaluation = evaluation_raw

    return ("Pipeline finished successfully.", notes, flashcards, evaluation)


def load_last_results():
    """Load and return the last generated notes, flashcards, and evaluation from disk if they exist."""
    notes = _read(NOTES_PATH)
    flashcards = _read(FLASHCARDS_PATH)
    evaluation_raw = _read(EVALUATION_PATH)
    
    if notes or flashcards or evaluation_raw:
        status = "Loaded last generated results from workspace."
        try:
            evaluation = json.dumps(json.loads(evaluation_raw), indent=2)
        except (json.JSONDecodeError, TypeError):
            evaluation = evaluation_raw
        return status, notes, flashcards, evaluation
    return "Ready to process lecture. No prior runs found.", "", "", ""


with gr.Blocks(title="EduAgent-OS") as demo:
    gr.Markdown("# EduAgent-OS")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        url_input = gr.Textbox(
            label="YouTube URL or local audio path",
            placeholder="https://www.youtube.com/watch?v=...",
            scale=4,
        )
        submit_btn = gr.Button("Process Lecture", variant="primary", scale=1)

    with gr.Accordion("API Configuration", open=False):
        api_key_input = gr.Textbox(
            label="Gemini API Key (optional if the host already has one configured)",
            placeholder="Enter your Gemini API key",
            type="password",
            # SECURITY: never prefill from the environment - the value would be
            # serialized into the page and visible to every visitor's browser.
            value="",
        )

    status_output = gr.Textbox(label="Status", interactive=False)

    with gr.Tabs():
        with gr.Tab("Study Notes"):
            notes_output = gr.Markdown()
        with gr.Tab("Flashcards"):
            flashcards_output = gr.Markdown()
        with gr.Tab("Evaluation Report"):
            evaluation_output = gr.Code(language="json")

    submit_btn.click(
        fn=process_lecture,
        inputs=[url_input, api_key_input],
        outputs=[status_output, notes_output, flashcards_output, evaluation_output],
    )
    url_input.submit(
        fn=process_lecture,
        inputs=[url_input, api_key_input],
        outputs=[status_output, notes_output, flashcards_output, evaluation_output],
    )

    demo.load(
        fn=load_last_results,
        outputs=[status_output, notes_output, flashcards_output, evaluation_output]
    )

demo.queue(max_size=5)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
