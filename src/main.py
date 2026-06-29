import asyncio
import os
from google import genai
from google.genai import types
from src.tools.media_fetcher import process_input_source
from src.tools.transcriber import transcribe_audio_file
from src.schema import LectureEvaluation

async def run_educational_pipeline(input_source: str):
    print(f"\n[Orchestrator] Task initialized: Processing '{input_source}'")
    
    # 1. Fetch the media file locally or via remote download
    audio_path = process_input_source(input_source)
    
    # 2. Local transcription using faster-whisper
    segments = transcribe_audio_file(audio_path)
    raw_transcript_text = "\n".join([f"[{s['start']:.1f}s - {s['end']:.1f}s]: {s['text']}" for s in segments])
    
    # Cache the raw transcript to the workspace volume
    transcript_log_path = "/app/workspace/transcript.txt"
    with open(transcript_log_path, "w") as f:
        f.write(raw_transcript_text)

    # 3. Native initialization compliant with both legacy and new AQ.Ab token standards
    api_key_env = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key_env)

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

    print("[Orchestrator] Running specialized generation workloads in parallel...")
    
    # Execute generation loops concurrently using gemini-2.5-flash
    notes_task = asyncio.to_thread(
        client.models.generate_content,
        model='gemini-2.5-flash',
        contents=synthesis_prompt
    )
    
    flash_task = asyncio.to_thread(
        client.models.generate_content,
        model='gemini-2.5-flash',
        contents=taxonomy_prompt
    )
    
    notes_response, flash_response = await asyncio.gather(notes_task, flash_task)

    # Write output artifacts back to your host's shared workspace folder
    with open("/app/workspace/notes.md", "w") as f:
        f.write(notes_response.text)
        
    with open("/app/workspace/flashcards.md", "w") as f:
        f.write(flash_response.text)

    # 4. Verification and Evaluation Pass (Day 4 Rubric Alignment)
    print("[Orchestrator] Activating semantic and structural verification layer...")
    
    eval_prompt = (
        f"Critically assess the generated study notes and flashcards against the original transcript. "
        f"Audit for factual inconsistencies, omissions, or ungrounded claims.\n\n"
        f"Original Transcript:\n{raw_transcript_text}\n\n"
        f"Generated Notes:\n{notes_response.text}\n\n"
        f"Generated Flashcards:\n{flash_response.text}"
    )

    # Force a structured response adhering strictly to our Pydantic schema
    verification_response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=eval_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=LectureEvaluation,
        ),
    )

    print(f"\n[Evaluation Report Results]:\n{verification_response.text}")
    print("\n[Orchestrator] Execution finished successfully.")

if __name__ == "__main__":
    target_input = os.getenv("LECTURE_TARGET", "https://www.youtube.com/watch?v=X6eGCO_5KOA")
    asyncio.run(run_educational_pipeline(target_input))
