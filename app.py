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
import tempfile

import gradio as gr

from src.config import get_workspace_paths
from src.main import run_educational_pipeline
from src.tools.pdf_generator import compile_markdown_to_pdf

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
    """Run the pipeline in an isolated per-request workspace.

    Returns (status, notes, flashcards, evaluation_json, pdf_file, session_dir).
    Each request gets its own temp workspace so concurrent visitors on a
    shared Space never see or overwrite each other's artifacts.
    """
    empty = ("", "", "", None, "")

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

    session_dir = tempfile.mkdtemp(prefix="eduagent-")
    paths = get_workspace_paths(session_dir)

    progress(0.05, desc="Initializing pipeline...")
    try:
        def on_progress(fraction: float, desc: str):
            progress(fraction, desc=desc)

        await run_educational_pipeline(
            input_source.strip(),
            progress_callback=on_progress,
            api_key=effective_key,
            workspace_dir=session_dir,
        )
    except Exception as e:
        return (f"Error: {type(e).__name__}: {e}", *empty)

    progress(0.95, desc="Loading generated artifacts...")

    notes = _read(paths["notes"])
    flashcards = _read(paths["flashcards"])
    evaluation_raw = _read(paths["evaluation"])
    try:
        evaluation = json.dumps(json.loads(evaluation_raw), indent=2)
    except (json.JSONDecodeError, TypeError):
        evaluation = evaluation_raw

    return ("Pipeline finished successfully.", notes, flashcards, evaluation, None, session_dir)


def export_pdf_action(session_dir: str):
    """Compiles this session's notes into a PDF and returns status and the file path."""
    if not session_dir:
        return ("Error: No study notes generated yet. Please process a lecture first.", None)

    paths = get_workspace_paths(session_dir)
    if not os.path.exists(paths["notes"]):
        return ("Error: No study notes generated yet. Please process a lecture first.", None)

    try:
        compile_markdown_to_pdf(paths["notes"], paths["pdf"])
        return ("PDF successfully generated!", paths["pdf"])
    except Exception as e:
        return (f"Error generating PDF: {type(e).__name__}: {e}", None)


theme = gr.themes.Soft(
    primary_hue="orange",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "sans-serif"],
).set(
    # Dark Midnight Backgrounds
    body_background_fill="#070a13",
    body_background_fill_dark="#070a13",
    block_background_fill="#0f172a",
    block_background_fill_dark="#0f172a",
    
    # Border overrides
    block_border_color="#1e293b",
    block_border_width="1px",
    
    # Premium Glowing Orange Buttons
    button_primary_background_fill="linear-gradient(135deg, #ff7e5f, #feb47b)",
    button_primary_background_fill_hover="linear-gradient(135deg, #feb47b, #ff7e5f)",
    button_primary_text_color="#ffffff",
    
    # Inputs
    input_background_fill="#1e293b",
    input_border_color="#334155",
)

with gr.Blocks(title="EduAgent-OS", theme=theme, js="() => document.documentElement.classList.add('dark')") as demo:
    gr.HTML("""
    <div style="text-align: center; padding: 2rem 1rem; background: linear-gradient(135deg, rgba(255, 126, 95, 0.1) 0%, rgba(254, 180, 123, 0.05) 100%); border: 1px solid rgba(255, 126, 95, 0.2); border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4); backdrop-filter: blur(5px);">
        <div style="display: inline-flex; align-items: center; gap: 0.5rem; background: rgba(255, 126, 95, 0.15); border: 1px solid rgba(255, 126, 95, 0.3); border-radius: 30px; padding: 0.25rem 0.75rem; margin-bottom: 1rem;">
            <span style="font-size: 0.75rem; font-weight: 700; color: #ffb4a2; letter-spacing: 0.05em; text-transform: uppercase;">🎓 Capstone Showcase -- Agents for Good</span>
        </div>
        <h1 style="font-size: 3rem; font-weight: 900; background: linear-gradient(90deg, #ff7e5f, #feb47b); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 0.5rem 0; letter-spacing: -0.02em;">EduAgent-OS</h1>
        <p style="font-size: 1.1rem; color: #94a3b8; max-width: 650px; margin: 0 auto; line-height: 1.6;">
            Transform any YouTube lecture or local audio into structured study notes, flashcards, and a hallucination audit report -- fully local, fully verified.
        </p>
        <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 1.5rem; flex-wrap: wrap;">
            <span style="font-size: 0.8rem; background: #0f172a; border: 1px solid #1e293b; color: #cbd5e1; border-radius: 6px; padding: 0.25rem 0.6rem;">🎙️ Offline Whisper Model</span>
            <span style="font-size: 0.8rem; background: #0f172a; border: 1px solid #1e293b; color: #cbd5e1; border-radius: 6px; padding: 0.25rem 0.6rem;">⚡ Parallel Synthesis</span>
            <span style="font-size: 0.8rem; background: #0f172a; border: 1px solid #1e293b; color: #cbd5e1; border-radius: 6px; padding: 0.25rem 0.6rem;">🔍 Factual Auditing</span>
        </div>
    </div>
    """)
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
    # Per-visitor session workspace path; keeps concurrent users isolated.
    session_state = gr.State("")

    with gr.Tabs():
        with gr.Tab("Study Notes"):
            with gr.Row():
                pdf_button = gr.Button("Export Notes to PDF", variant="secondary", scale=1)
                pdf_file = gr.File(label="Download PDF Study Guide", scale=2)
            notes_output = gr.Markdown()
        with gr.Tab("Flashcards"):
            flashcards_output = gr.Markdown()
        with gr.Tab("Evaluation Report"):
            evaluation_output = gr.Code(language="json")

    submit_btn.click(
        fn=process_lecture,
        inputs=[url_input, api_key_input],
        outputs=[status_output, notes_output, flashcards_output, evaluation_output, pdf_file, session_state],
    )
    url_input.submit(
        fn=process_lecture,
        inputs=[url_input, api_key_input],
        outputs=[status_output, notes_output, flashcards_output, evaluation_output, pdf_file, session_state],
    )

    pdf_button.click(
        fn=export_pdf_action,
        inputs=[session_state],
        outputs=[status_output, pdf_file]
    )

demo.queue(max_size=5)

if __name__ == "__main__":
    # Gradio 6 moved theme from the Blocks constructor to launch()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        theme=theme,
    )
