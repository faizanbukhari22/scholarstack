# EduAgent-OS: Containerized Multi-Agent Pipeline for Autonomous Lecture Intelligence

**Subtitle:** From any YouTube lecture URL to structured study notes, flashcards, and a hallucination audit report -- in under 3 minutes, with zero cloud transcription costs and zero manual effort.

**Track:** Agents for Good

---

## The Problem

Students and self-learners spend enormous amounts of time on a task that adds no conceptual value: manually re-watching lectures, pausing to take notes, and building flashcard decks by hand. This is not a learning problem -- it is a logistics problem. The actual understanding happens when reading and reviewing the material, not when copying it down.

Existing tools fail this use case in one of three ways. They either require manual input from the user, send audio to expensive cloud transcription APIs that accumulate per-minute costs, or produce unverified outputs with no quality guarantee. A student who trusts an AI-generated summary that hallucinated a definition has been actively harmed, not helped.

EduAgent-OS was built to solve all three failure modes at once: fully automated, fully local transcription, and a structured verification pass that audits the output for factual consistency before the student ever reads it.

---

## The Solution

EduAgent-OS is a containerized, production-grade multi-agent pipeline. A user provides a YouTube lecture URL or a local audio file. The system handles everything else:

1. Downloads and extracts the audio stream locally using yt-dlp
2. Transcribes it on-device using faster-whisper -- no audio leaves the container
3. Dispatches two specialized AI agents in parallel to synthesize study materials
4. Runs a structured verification agent that scores the output for hallucinations and factual drift; if the audit flags ungrounded claims or low consistency, the findings are fed back to the synthesis agents for a revision pass and the revised outputs are re-audited before final delivery

The result is four files written to the user's workspace folder:

- `transcript.txt` -- full timestamped lecture transcript
- `notes.md` -- hierarchical Markdown study notes
- `flashcards.md` -- Anki-compatible Q&A flashcard table
- `evaluation.json` -- structured hallucination audit report with consistency scores

The pipeline is fully containerized via Docker and Compose, runs on any machine with Docker installed, and requires only a Gemini API key.

---

## Why Agents

This problem is not solvable with a single prompt. Three distinct specialized behaviors are required, and they must happen in a specific sequence with a verification pass at the end.

A single generalist prompt would produce mediocre notes with no specialization, no parallel efficiency, and no quality gate. By separating the work into agents with distinct roles and running them concurrently, EduAgent-OS produces better output faster and then validates it before delivery.

The two synthesis agents have fundamentally different jobs. The Academic Synthesis Specialist is instructed to think like a professor organizing a lecture into study materials. The Educational Taxonomist is instructed to think like an exam designer extracting testable facts. Neither role produces good output from the other's instruction set. The verification agent is a third role entirely: a critical auditor comparing output against source rather than generating anything new.

---

## Architecture

The system is organized into five isolated, deterministic layers:

### Layer 1 -- Ingestion (media_fetcher.py)

Accepts a YouTube URL or local file path. For URLs, yt-dlp downloads and post-processes the audio stream to a normalized 192kbps MP3 using FFmpeg. The fetcher checks for a cached MP3 by extracting the video ID from the URL before attempting any network call, so repeat runs skip the download entirely.

### Layer 2 -- Local Acoustic Transcription (transcriber.py)

Initializes a faster-whisper model (base, int8 quantized, CPU) from a path baked directly into the Docker image during build time. The model is pre-downloaded at `docker build` into `/app/models` and loaded at runtime from that path via the `WHISPER_MODEL_PATH` environment variable. No HuggingFace network calls occur at runtime. The model produces timestamped text segments cached to `workspace/transcript.txt`.

### Layer 3 -- Parallel Agent Orchestration (main.py)

The core multi-agent loop. Two Gemini agents are dispatched concurrently using `asyncio.gather` and `asyncio.to_thread` against `gemini-2.5-flash` via the official google-genai SDK:

**Academic Synthesis Specialist** -- prompt-instructed to organize the transcript into hierarchical Markdown with clear `# Summary` and `## Methodology` structure, suitable for revision.

**Educational Taxonomist** -- prompt-instructed to extract critical terminology, formulas, and equations into a Q&A table matrix compatible with Anki import.

Both agents receive the same timestamped transcript and run simultaneously. Neither blocks the other. Their outputs are written to `workspace/notes.md` and `workspace/flashcards.md` once both complete.

### Layer 4 -- Structured Verification with Correction Loop (schema.py)

A third Gemini call audits both outputs against the original transcript. The agent is constrained to respond strictly in JSON conforming to the `LectureEvaluation` Pydantic v2 model, enforced via `response_schema` and `response_mime_type="application/json"`:

```python
class LectureEvaluation(BaseModel):
    factual_consistency_score: float
    summary_quality_score: float
    hallucination_detected: bool
    hallucinated_claims: list[str]
    missing_critical_terms: list[str]
    key_concepts_covered: list[str]
```

This is not optional post-processing -- it is a quality gate. If the audit reports hallucinated claims or a factual consistency score below a configurable threshold (default 0.85), the pipeline feeds the audit findings back to both synthesis agents for one revision pass and re-audits the revised outputs before delivery. The judge model is separately configurable (e.g. `gemini-2.5-pro`) to reduce self-grading bias. The final verification result is written to `workspace/evaluation.json` as a persistent audit artifact: a student can open this file and see exactly which claims were flagged as ungrounded and which transcript terms were omitted.

### Layer 5 -- MCP Server (mcp_server.py)

The entire pipeline is also exposed as an MCP (Model Context Protocol) server, so any MCP-compatible client can drive EduAgent-OS as a composable tool rather than a standalone script (see Key Concepts below).

---

## Key Concepts Demonstrated

**Multi-Agent System:** Two specialized sub-agents (Academic Synthesis Specialist and Educational Taxonomist) run in parallel via `asyncio.gather` with `asyncio.to_thread`, followed by a third independent verification agent whose audit can trigger a revision pass -- a closed agentic loop, not a linear script. Each agent has a distinct role, distinct instruction set, and distinct output artifact.

**MCP Server:** The full pipeline is also exposed as an MCP (Model Context Protocol) server (`src/mcp_server.py`, built with the official Python MCP SDK / FastMCP) with six tools: `process_lecture`, `get_notes`, `get_flashcards`, `get_evaluation`, `get_transcript`, and `export_pdf`. Claude Desktop, Claude Code, or any MCP-compatible client can drive EduAgent-OS as a composable tool rather than a standalone script.

**Security Features:** The Gemini API key is never stored in code or committed to version control. It is passed into the container exclusively through Docker Compose's `environment` block, sourced from a local `.env` file that is explicitly excluded by `.gitignore`. Audio data never leaves the container boundary -- all transcription is performed locally by faster-whisper. The workspace directory containing user data is also gitignored. The ingestion surface is hardened for public deployment: remote URLs are validated against a YouTube host whitelist before any network call (SSRF prevention), local file paths are sandboxed to the workspace directory (no arbitrary container file reads), transcript content is delimited and marked as untrusted data in every prompt (prompt-injection mitigation), the container runs as a non-root user, and the web demo isolates each request in its own temporary workspace so concurrent users can never see each other's data.

**Deployability:** The project is fully containerized via Docker and Docker Compose. A `python:3.11-slim` base image with explicit platform handling ensures reproducibility across ARM64 and AMD64 architectures. A GitHub Actions CI pipeline (`ci.yml`) runs flake8 static analysis on every push to `main`, with a hard-fail pass for syntax errors and undefined names.

---

## Technical Implementation Highlights

**Model baking for zero-latency cold starts.** The faster-whisper base model is pre-downloaded into the image layer at build time using `download_root='/app/models'`. At runtime, the container loads model weights from disk in under a second with no network activity. This reduced cold-start time from 3-8 minutes to under 30 seconds on repeat runs.

**Download caching for repeat runs.** The media fetcher extracts the YouTube video ID from the URL using regex and checks whether `{video_id}.mp3` already exists in the workspace before invoking yt-dlp. Cache hits are logged and the download is skipped entirely.

**Typed output enforcement.** Rather than parsing free-text model responses, the verification layer uses `response_schema=LectureEvaluation` to force Gemini to emit structured JSON that Pydantic validates at the schema level. This makes the audit result machine-readable and eliminates parsing fragility.

**Portable containerization.** The Dockerfile uses `ARG BUILDPLATFORM` with `--platform=${BUILDPLATFORM:-linux/amd64}` rather than hardcoding `linux/arm64`, ensuring the image builds natively on both Apple Silicon and AMD64 CI runners without emulation overhead.

---

## Live Execution Results

Running `docker-compose up --build` on the Kaggle capstone overview lecture produced the following:

```
[Fetcher] Cache hit -- skipping download: X6eGCO_5KOA.mp3
[Transcriber] Attempting to load Whisper model (base) from baked cache: /app/models...
[Pipeline Progress] 55%: Generating structured study notes and flashcards in parallel...
[Pipeline Progress] 75%: Running factual audit and hallucination checks...

[Evaluation Report Results]:
{
  "factual_consistency_score": 0.95,
  "summary_quality_score": 0.98,
  "hallucination_detected": false,
  "hallucinated_claims": [],
  "missing_critical_terms": ["ChatGPT prompts", "Hugging Face", "ADK"],
  "key_concepts_covered": [
    "Capstone Project purpose and scope",
    "AASN tools, skills, security, deployment",
    "Badge and certificate eligibility",
    "Four project tracks (Good, Business, Concierge, Freestyle)",
    "Submission deadline (July 6, 2026)",
    "Evaluation scoring breakdown (30 marks concept, 70 marks code)"
  ]
}
```

The auditor confirmed every claim in the outputs is grounded in the transcript (`hallucinated_claims` is empty) while still surfacing three transcript terms the synthesis agents omitted -- useful signal for the student even when the quality gate passes. Had any ungrounded claim been found, or had consistency dropped below the 0.85 threshold, the pipeline would have automatically run a revision pass and re-audited before delivering the files.

---

## Project Journey

The project began as a simple Python script calling the Gemini API with a hardcoded transcript. The first real engineering challenge was eliminating the dependency on cloud transcription -- sending lecture audio to an external API creates both cost and privacy concerns that defeat the purpose of a student productivity tool.

Integrating faster-whisper locally solved the transcription problem but introduced a new one: cold-start latency. Every container restart triggered a fresh model download from HuggingFace. The fix was baking the model weights into the Docker image layer at build time, which turned a 5-8 minute wait into a sub-second load from disk.

The parallel agent architecture came from recognizing that sequential prompting was slow and the two synthesis tasks were entirely independent. Switching to `asyncio.gather` with thread-based dispatch cut the generation phase to the time of the slower agent rather than the sum of both.

The verification layer was added last, after realizing that an unverified AI-generated summary is potentially worse than no summary at all. Using Pydantic's `response_schema` enforcement meant the audit result is always a typed, machine-readable object rather than a paragraph that needs to be interpreted. The final step was turning that audit from a passive report into an active gate: flagged outputs are revised once with the audit findings as feedback and re-audited, closing the agent loop.

---

## Setup and Reproduction

**Requirements:** Docker, Docker Compose v2.0+, Google AI Studio API key.

```bash
git clone https://github.com/faizanbukhari22/eduagent-os.git
cd eduagent-os
echo "GEMINI_API_KEY=your_key_here" > .env
docker-compose up --build
```

To process a different lecture:
```bash
LECTURE_TARGET="https://www.youtube.com/watch?v=your_video" docker-compose up
```

Output files appear in `workspace/` on the host machine after the container exits.

---

## Conclusion

EduAgent-OS demonstrates that multi-agent systems are most valuable when the problem genuinely requires distinct specialized behaviors operating in parallel with a verification pass at the end. The architecture directly mirrors how good human study groups work: one person focuses on narrative structure, another on extracting testable facts, and a third checks both against the source. The difference is that EduAgent-OS does this in under 3 minutes for any lecture on the internet.
