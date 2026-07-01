import asyncio
import os
from typing import Callable, Optional
from google import genai
from google.genai import types
from src.config import (
    TRANSCRIPT_PATH,
    NOTES_PATH,
    FLASHCARDS_PATH,
    EVALUATION_PATH,
)
from src.tools.media_fetcher import process_input_source
from src.tools.transcriber import transcribe_audio_file
from src.schema import LectureEvaluation

import random

async def generate_content_with_retry(
    client,
    model: str,
    contents,
    config=None,
    max_retries: int = 5,
    initial_delay: float = 2.0
):
    """Call generate_content with exponential backoff on 429 Rate Limit/Resource Exhausted errors."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return await asyncio.to_thread(
                client.models.generate_content,
                model=model,
                contents=contents,
                config=config
            )
        except Exception as e:
            err_msg = str(e).lower()
            is_rate_limit = (
                "429" in err_msg
                or "resource_exhausted" in err_msg
                or "resource exhausted" in err_msg
                or "rate limit" in err_msg
                or "quota" in err_msg
            )
            if is_rate_limit and attempt < max_retries - 1:
                # Add jitter to avoid thundering herd problem
                jitter = random.uniform(0.5, 1.5)
                sleep_time = delay * jitter
                print(f"[Gemini API] Rate limit hit: {e}. Retrying in {sleep_time:.1f}s... (Attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(sleep_time)
                delay *= 2.0
            else:
                raise e

async def run_educational_pipeline(
    input_source: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    api_key: Optional[str] = None,
) -> str:
    """Run the full ingestion -> transcription -> synthesis -> verification pipeline.

    Returns the raw JSON text of the structured LectureEvaluation report so
    callers (e.g. the MCP server) can hand it back without re-reading disk.
    """
    def report_progress(fraction: float, description: str):
        if progress_callback:
            try:
                progress_callback(fraction, description)
            except Exception as e:
                print(f"[Orchestrator] Warning: Progress callback error: {e}")
        else:
            print(f"[Pipeline Progress] {int(fraction * 100)}%: {description}")

    report_progress(0.05, "Task initialized: Processing input source...")

    # 1. Fetch the media file locally or via remote download
    report_progress(0.10, "Fetching audio stream (downloading or resolving cache)...")
    audio_path = process_input_source(input_source)

    # 2. Local transcription using faster-whisper (Returns pre-formatted string payload)
    report_progress(0.30, "Transcribing audio locally via Whisper...")
    raw_transcript_text = transcribe_audio_file(audio_path)

    # Cache the raw transcript to the workspace volume
    with open(TRANSCRIPT_PATH, "w") as f:
        f.write(raw_transcript_text)

    # 3. Initialize the GenAI client. An explicitly passed key takes priority
    # so shared frontends (e.g. the Gradio demo) never mutate os.environ.
    target_key = api_key or os.getenv("GEMINI_API_KEY")
    if not target_key:
        raise ValueError(
            "Missing GEMINI_API_KEY. Set it in the environment or pass api_key."
        )
    client = genai.Client(api_key=target_key)

    # Define specialized agent instructions
    synthesis_prompt = (
        "You are an Academic Synthesis Specialist. Organize the following transcript "
        "into comprehensive, highly structured Markdown study notes using clear hierarchy (# Summary, ## Methodology).\n\n"
        f"Source Transcript:\n{raw_transcript_text}"
    )
    
    taxonomy_prompt = (
        "You are an Educational Taxonomist. Extract all critical terminology, formulas, "
        "and mathematical equations from this transcript into a Q&A table matrix compatible with Anki.\n\n"
        f"Source Transcript:\n{raw_transcript_text}"
    )

    report_progress(0.60, "Generating structured study notes and flashcards in parallel...")
    
    # Execute generation loops concurrently using gemini-2.5-flash with retry logic
    notes_task = generate_content_with_retry(
        client,
        model='gemini-2.5-flash',
        contents=synthesis_prompt
    )
    
    flash_task = generate_content_with_retry(
        client,
        model='gemini-2.5-flash',
        contents=taxonomy_prompt
    )
    
    notes_response, flash_response = await asyncio.gather(notes_task, flash_task)

    # Write output artifacts back to your host's shared workspace folder
    with open(NOTES_PATH, "w") as f:
        f.write(notes_response.text)

    with open(FLASHCARDS_PATH, "w") as f:
        f.write(flash_response.text)

    # 4. Verification and Evaluation Pass (Day 4 Rubric Alignment)
    report_progress(0.85, "Running factual audit and hallucination checks...")
    
    eval_prompt = (
        "Critically assess the generated study notes and flashcards against the original transcript. "
        "Audit for factual inconsistencies, omissions, or ungrounded claims.\n\n"
        f"Original Transcript:\n{raw_transcript_text}\n\n"
        f"Generated Notes:\n{notes_response.text}\n\n"
        f"Generated Flashcards:\n{flash_response.text}"
    )

    # Force a structured response adhering strictly to our Pydantic schema with retry logic
    verification_response = await generate_content_with_retry(
        client,
        model='gemini-2.5-flash',
        contents=eval_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=LectureEvaluation,
        ),
    )

    with open(EVALUATION_PATH, "w") as f:
        f.write(verification_response.text)

    print(f"\n[Evaluation Report Results]:\n{verification_response.text}")
    
    report_progress(1.00, "EduAgent-OS run completed successfully.")

    return verification_response.text

if __name__ == "__main__":
    target_input = os.getenv("LECTURE_TARGET", "https://www.youtube.com/watch?v=X6eGCO_5KOA")
    asyncio.run(run_educational_pipeline(target_input))
