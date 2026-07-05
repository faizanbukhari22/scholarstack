#!/usr/bin/env python3
"""
Gradio frontend for ScholarStack.

Serves the containerized pipeline (ingestion, local transcription, parallel
Gemini synthesis, and structured hallucination verification) as an
interactive web demo.
"""

import json
import os

import gradio as gr

from src.config import get_lecture_paths, get_env, LIBRARY_DIR
from src.main import run_educational_pipeline
from src.tools.media_fetcher import get_or_create_lecture_dir
from src.tools.pdf_generator import compile_markdown_to_pdf


def _sanitize_error(e: Exception) -> str:
    """Strip API keys from exception messages before showing them to users."""
    msg = str(e)
    api_key = get_env("GEMINI_API_KEY")
    if api_key and api_key in msg:
        msg = msg.replace(api_key, "[REDACTED]")
    # Also redact partial key fragments the SDK may include
    if api_key and len(api_key) > 8:
        fragment = api_key[:8]
        msg = msg.replace(fragment, "[REDACTED]")
    return f"{type(e).__name__}: {msg}"

DESCRIPTION = (
    "Paste a YouTube lecture URL below. ScholarStack transcribes it locally with "
    "faster-whisper, dispatches two parallel Gemini agents to write structured "
    "study notes and Anki-style flashcards, then runs a verification pass that "
    "audits both outputs for factual consistency and hallucinations.\n\n"
    "Transcription and generation run on CPU, so processing can take a few "
    "minutes depending on lecture length and the Space's hardware tier."
)


def _read(path: str) -> str:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def get_library_choices():
    """Scan the library directory and return list of completed lectures."""
    import glob
    choices = []
    folders = glob.glob(os.path.join(LIBRARY_DIR, "*__*"))
    for folder in folders:
        meta_path = os.path.join(folder, "meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("status") == "complete":
                    title = meta.get("title", "Unknown Title")
                    choices.append((f"{title} ({meta.get('id')})", folder))
            except Exception:
                pass
    choices.sort(key=lambda x: x[0])
    return choices


def update_library_dropdown():
    """Return a fresh Dropdown component filled with completed library choices."""
    return gr.Dropdown(choices=get_library_choices())


async def process_lecture(
    input_source: str,
    api_key: str = "",
    force_mode: str = "None",
    progress=gr.Progress(track_tqdm=False)
):
    """Run the pipeline in the resolved persistent lecture workspace folder.

    Returns (status, notes, flashcards, evaluation_json, pdf_file, lecture_dir).
    """
    empty = ("", "", "", None, "")

    if not input_source or not input_source.strip():
        return ("Please paste a YouTube URL or local file path.", *empty)

    effective_key = (api_key or "").strip() or get_env("GEMINI_API_KEY")
    if not effective_key:
        return (
            "Error: No Gemini API key available. "
            "Provide one in the configuration section, or configure GEMINI_API_KEY on the host.",
            *empty,
        )

    progress(0.05, desc="Resolving lecture directory...")
    try:
        lecture_dir, meta = get_or_create_lecture_dir(input_source.strip(), LIBRARY_DIR)
        paths = get_lecture_paths(lecture_dir)
    except Exception as e:
        return (f"Error resolving source: {_sanitize_error(e)}", *empty)

    force = (force_mode == "Gemini Only (Regenerate Study Materials)")
    force_all = (force_mode == "Full Run (Redownload & Retranscribe)")

    progress(0.10, desc="Initializing pipeline...")
    try:
        def on_progress(fraction: float, desc: str):
            progress(fraction, desc=desc)

        await run_educational_pipeline(
            input_source.strip(),
            progress_callback=on_progress,
            api_key=effective_key,
            force=force,
            force_all=force_all,
        )
    except Exception as e:
        return (f"Error: {_sanitize_error(e)}", *empty)

    progress(0.95, desc="Loading generated artifacts...")

    notes = _read(paths["notes"])
    flashcards = _read(paths["flashcards"])
    evaluation_raw = _read(paths["evaluation"])
    try:
        evaluation = json.dumps(json.loads(evaluation_raw), indent=2)
    except (json.JSONDecodeError, TypeError):
        evaluation = evaluation_raw

    # Check if PDF already exists
    pdf_file = paths["pdf"] if os.path.exists(paths["pdf"]) else None

    return ("Pipeline finished successfully.", notes, flashcards, evaluation, pdf_file, lecture_dir)


def load_library_lecture(lecture_dir: str):
    """Load and return generated artifacts from a selected library lecture folder."""
    if not lecture_dir or not os.path.exists(lecture_dir):
        return ("Please select a valid lecture.", "", "", "", None, "")

    # M1: Guard against path traversal — only allow paths inside LIBRARY_DIR.
    real_dir = os.path.realpath(lecture_dir)
    real_lib = os.path.realpath(LIBRARY_DIR)
    if not (real_dir == real_lib or real_dir.startswith(real_lib + os.sep)):
        return ("Error: Invalid lecture directory.", "", "", "", None, "")

    paths = get_lecture_paths(lecture_dir)
    notes = _read(paths["notes"])
    flashcards = _read(paths["flashcards"])
    evaluation_raw = _read(paths["evaluation"])
    try:
        evaluation = json.dumps(json.loads(evaluation_raw), indent=2)
    except (json.JSONDecodeError, TypeError):
        evaluation = evaluation_raw

    pdf_file = paths["pdf"] if os.path.exists(paths["pdf"]) else None

    return ("Lecture loaded successfully.", notes, flashcards, evaluation, pdf_file, lecture_dir)


def export_pdf_action(session_dir: str):
    """Compiles this session's notes into a PDF and returns status and the file path."""
    if not session_dir:
        return ("Error: No study notes generated yet. Please process a lecture first.", None)

    # M1: Guard against path traversal.
    real_dir = os.path.realpath(session_dir)
    real_lib = os.path.realpath(LIBRARY_DIR)
    if not (real_dir == real_lib or real_dir.startswith(real_lib + os.sep)):
        return ("Error: Invalid session directory.", None)

    paths = get_lecture_paths(session_dir)

    # M3: Check meta.json status before allowing PDF export.
    meta_path = paths["meta"]
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("status") != "complete":
                return (f"Error: Lecture is not fully processed (status: {meta.get('status', 'unknown')}). Run the pipeline first.", None)
        except Exception:
            pass

    if not os.path.exists(paths["notes"]):
        return ("Error: No study notes generated yet. Please process a lecture first.", None)

    try:
        compile_markdown_to_pdf(paths["notes"], paths["pdf"])
        return ("PDF successfully generated!", paths["pdf"])
    except Exception as e:
        return (f"Error generating PDF: {_sanitize_error(e)}", None)


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

with gr.Blocks(title="ScholarStack") as demo:
    gr.HTML("""
    <div style="text-align: center; padding: 2rem 1rem; background: linear-gradient(135deg, rgba(255, 126, 95, 0.1) 0%, rgba(254, 180, 123, 0.05) 100%); border: 1px solid rgba(255, 126, 95, 0.2); border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4); backdrop-filter: blur(5px);">
        <div style="display: inline-flex; align-items: center; gap: 0.5rem; background: rgba(255, 126, 95, 0.15); border: 1px solid rgba(255, 126, 95, 0.3); border-radius: 30px; padding: 0.25rem 0.75rem; margin-bottom: 1rem;">
            <span style="font-size: 0.75rem; font-weight: 700; color: #ffb4a2; letter-spacing: 0.05em; text-transform: uppercase;">🎓 Capstone Showcase -- Agents for Good</span>
        </div>
        <h1 style="font-size: 3rem; font-weight: 900; background: linear-gradient(90deg, #ff7e5f, #feb47b); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 0.5rem 0; letter-spacing: -0.02em;">ScholarStack</h1>
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
            scale=3,
        )
        force_mode_input = gr.Dropdown(
            label="Processing Mode",
            choices=["None", "Gemini Only (Regenerate Study Materials)", "Full Run (Redownload & Retranscribe)"],
            value="None",
            scale=2,
        )
        submit_btn = gr.Button("Process Lecture", variant="primary", scale=1)

    with gr.Accordion("API Configuration", open=False):
        api_key_input = gr.Textbox(
            label="Gemini API Key (optional if the host already has one configured)",
            placeholder="Enter your Gemini API key",
            type="password",
            value="",
        )

    status_output = gr.Textbox(label="Status", interactive=False)
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
        with gr.Tab("Lecture Library"):
            with gr.Row():
                library_dropdown = gr.Dropdown(
                    label="Select Processed Lecture",
                    choices=get_library_choices(),
                    interactive=True,
                    scale=4,
                )
                refresh_btn = gr.Button("🔄 Refresh", scale=1)
                load_btn = gr.Button("📂 Load Data", variant="primary", scale=1)

    # Event handlers
    submit_btn.click(
        fn=process_lecture,
        inputs=[url_input, api_key_input, force_mode_input],
        outputs=[status_output, notes_output, flashcards_output, evaluation_output, pdf_file, session_state],
    ).then(
        fn=update_library_dropdown,
        inputs=[],
        outputs=[library_dropdown],
    )
    
    url_input.submit(
        fn=process_lecture,
        inputs=[url_input, api_key_input, force_mode_input],
        outputs=[status_output, notes_output, flashcards_output, evaluation_output, pdf_file, session_state],
    ).then(
        fn=update_library_dropdown,
        inputs=[],
        outputs=[library_dropdown],
    )

    pdf_button.click(
        fn=export_pdf_action,
        inputs=[session_state],
        outputs=[status_output, pdf_file]
    )

    refresh_btn.click(
        fn=update_library_dropdown,
        inputs=[],
        outputs=[library_dropdown],
    )

    load_btn.click(
        fn=load_library_lecture,
        inputs=[library_dropdown],
        outputs=[status_output, notes_output, flashcards_output, evaluation_output, pdf_file, session_state],
    )

demo.queue(max_size=5)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        theme=theme,
        js="() => document.documentElement.classList.add('dark')",
    )
