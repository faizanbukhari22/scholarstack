---
title: EduAgent-OS
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
suggested_hardware: cpu-basic
short_description: Turn any YouTube lecture into study notes, flashcards, and a hallucination audit
---

# EduAgent-OS

**Track: Agents for Good**

A containerized, production-grade multi-agent pipeline that transforms any YouTube lecture or local audio file into structured study notes, Anki-compatible flashcards, and a structured JSON hallucination audit report - fully automated, with fully local transcription and zero cloud transcription costs.

---

## Problem Statement

Students and self-learners spend a significant amount of time manually re-watching lectures, pausing to take notes, and building flashcard decks - work that is repetitive and time-consuming but adds no conceptual value. Existing tools either require manual input, send audio to expensive cloud APIs, or produce unverified outputs with no quality guarantee.

EduAgent-OS solves this by running a fully automated, containerized agent pipeline: it ingests a lecture URL or local audio file, transcribes it locally using a hardware-native ML model, dispatches two specialized AI agents in parallel to synthesize study materials, and then runs a structured verification pass that scores the output for factual consistency and hallucination detection before writing everything to disk.

---

## System Architecture

The runtime is split into five isolated, deterministic layers:

### Layer 1 - Ingestion (`src/tools/media_fetcher.py`)
Accepts a YouTube URL or a local file path. For remote URLs, `yt-dlp` downloads and extracts the best available audio stream, post-processing it to a normalized 192kbps `.mp3` via FFmpeg. For local files, it resolves the path against the mounted workspace volume and passes it forward. No audio ever leaves the container boundary.

### Layer 2 - Acoustic Transcription (`src/tools/transcriber.py`)
Initializes a `faster-whisper` model (base, int8 quantized, CPU) inside the container. The model produces timestamped text segments with start/end times for every dialogue block. The full transcript is cached to `workspace/transcript.txt` for reproducibility and debugging.

### Layer 3 - Parallel Agent Orchestration (`src/main.py`)
The core multi-agent loop. Two specialized Gemini agents are dispatched concurrently using `asyncio.gather` and `asyncio.to_thread`:

- **Academic Synthesis Specialist**: Organizes the transcript into hierarchical Markdown study notes with a `# Summary` and `## Methodology` structure, suitable for revision.
- **Educational Taxonomist**: Extracts critical terminology, formulas, and equations from the transcript into a Q&A table matrix compatible with Anki import.

Both agents run against `gemini-2.5-flash` via the official `google-genai` SDK. Their outputs are written to `workspace/notes.md` and `workspace/flashcards.md`.

### Layer 4 - Structured Verification with Correction Loop (`src/schema.py`)
After generation, a verification agent audits both outputs against the original transcript. It is forced to respond in structured JSON conforming to the `LectureEvaluation` Pydantic model, which captures:

| Field | Type | Description |
|---|---|---|
| `factual_consistency_score` | float (0-1) | How accurately the notes reflect the source transcript |
| `summary_quality_score` | float (0-1) | Clarity, structure, and completeness of the study notes |
| `hallucination_detected` | bool | Whether any claim cannot be grounded in the transcript |
| `hallucinated_claims` | list[str] | The specific ungrounded claims found in the outputs |
| `missing_critical_terms` | list[str] | Important concepts omitted from the outputs |
| `key_concepts_covered` | list[str] | Important concepts successfully captured |

The audit is a real quality gate, not a passive report: if `hallucination_detected` is true or `factual_consistency_score` falls below a configurable threshold (`EDUAGENT_MIN_CONSISTENCY`, default 0.85), the audit findings are fed back to both synthesis agents for a revision pass and the revised outputs are re-audited before delivery. The judge model is independently configurable via `EDUAGENT_JUDGE_MODEL` (e.g. `gemini-2.5-pro`) to reduce self-grading bias. The final result is printed to stdout and saved to `workspace/evaluation.json`.

### Layer 5 - MCP Server (`src/mcp_server.py`)
The entire pipeline is also exposed as an MCP (Model Context Protocol) server named `eduagent_mcp`, so any MCP-compatible client (Claude Desktop, Claude Code, or another agent) can call EduAgent-OS as a tool rather than running it as a standalone script. See [MCP Server](#mcp-server) below.

---

## Rubric Coverage

| Rubric Criterion | Implementation |
|---|---|
| **Multi-Agent Architecture** | Two specialized sub-agents (Academic Synthesis Specialist and Educational Taxonomist) run in parallel via `asyncio.gather` with `asyncio.to_thread` |
| **Structured Output Schema** | `LectureEvaluation` Pydantic model enforces typed JSON from Gemini via `response_schema` and `response_mime_type="application/json"` |
| **Security Features** | API key injected at runtime via Docker Compose environment variables; `.env` and `workspace/` gitignored; remote ingestion restricted to a YouTube host whitelist (SSRF prevention); local file access sandboxed to the workspace directory; untrusted transcript content delimited against prompt injection; container runs as a non-root user; per-request workspaces isolate concurrent web demo users |
| **Deployability & CI** | Fully containerized via Docker + Docker Compose; GitHub Actions CI gate runs flake8 static analysis on every push to `main` |
| **Tool Execution** | `yt-dlp` for media ingestion, `faster-whisper` for local ML transcription - both execute as tool calls inside the container boundary |
| **MCP Server** | `src/mcp_server.py` exposes the full pipeline as MCP tools (`process_lecture`, `get_notes`, `get_flashcards`, `get_evaluation`, `get_transcript`, `export_pdf`) via the official Python MCP SDK (FastMCP), callable from Claude Desktop, Claude Code, or any MCP client |
| **Live Deployment** | `app.py` Gradio frontend runs as the Docker image's default entrypoint, deployable directly to Hugging Face Spaces for a public, interactive demo |

---

## Quickstart

### Prerequisites

- Docker and Docker Compose v2.0+
- A Google AI Studio API key (`GEMINI_API_KEY`)

### 1. Clone and configure

```bash
git clone https://github.com/faizanbukhari22/eduagent-os.git
cd eduagent-os
```

Create a `.env` file in the project root (this file is gitignored and never committed):

```
GEMINI_API_KEY=your_api_key_here
```

### 2. Run the pipeline

```bash
docker-compose up --build
```

To process a different lecture, pass a custom URL:

```bash
LECTURE_TARGET="https://www.youtube.com/watch?v=your_video_id" docker-compose up --build
```

### 3. View the outputs

All output files are written to the `workspace/` folder on your host machine:

| File | Description |
|---|---|
| `workspace/transcript.txt` | Timestamped transcript of the full lecture |
| `workspace/notes.md` | Structured Markdown study notes |
| `workspace/flashcards.md` | Anki-compatible Q&A flashcard table |
| `workspace/evaluation.json` | Structured JSON hallucination audit report |

---

## Project Structure

```
eduagent-os/
├── app.py                   # Gradio frontend - Docker/HF Spaces default entrypoint
├── src/
│   ├── main.py              # Orchestration loop and async agent dispatch
│   ├── mcp_server.py        # FastMCP server exposing the pipeline as MCP tools
│   ├── config.py            # Shared WORKSPACE_DIR resolution (container + local)
│   ├── schema.py            # Pydantic LectureEvaluation structured output schema
│   └── tools/
│       ├── media_fetcher.py # URL/local file ingestion via yt-dlp (host whitelist + path sandbox)
│       ├── transcriber.py   # Local ML transcription via faster-whisper
│       └── pdf_generator.py # Markdown-to-PDF study guide compiler (fpdf2)
├── sample_output/           # Sample pipeline outputs from a real run
│   ├── transcript.txt
│   ├── notes.md
│   ├── flashcards.md
│   └── evaluation.json
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions CI gate (flake8 static analysis)
├── Dockerfile               # Container build spec (python:3.11-slim, ffmpeg, ARM64/AMD64)
├── docker-compose.yml       # Runtime orchestration with volume mount and env injection
├── requirements.txt         # Python dependencies
└── .gitignore               # Excludes .env, workspace/, __pycache__, *.mp3
```

---

## MCP Server

EduAgent-OS ships with an MCP (Model Context Protocol) server, `src/mcp_server.py`, built with the official Python MCP SDK (FastMCP). It exposes the pipeline as six callable tools instead of a single-shot script, so any MCP-compatible client can drive it directly.

### Tools

| Tool | Description |
|---|---|
| `process_lecture(input_source)` | Runs the full pipeline (ingest, transcribe, synthesize, verify) against a YouTube URL or local audio file. Returns the workspace paths and the parsed evaluation JSON. |
| `get_notes()` | Returns the Markdown study notes from the most recent run. |
| `get_flashcards()` | Returns the Anki-compatible flashcard table from the most recent run. |
| `get_evaluation()` | Returns the structured `LectureEvaluation` JSON from the most recent run. |
| `get_transcript()` | Returns the raw timestamped transcript from the most recent run. |
| `export_pdf()` | Compiles the latest study notes into a print-ready PDF with headers and page numbers. |

### Local setup

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_api_key_here
```

Run the server directly from the project root (stdio transport):

```bash
PYTHONPATH=. python -m src.mcp_server
```

`src/config.py` resolves the workspace directory automatically: it uses `/app/workspace` inside the Docker container and falls back to `<project_root>/workspace` for local/MCP execution, or honors an explicit `WORKSPACE_DIR` environment variable override.

Note: running `process_lecture` outside Docker requires `ffmpeg` on your `PATH` (needed by both `yt-dlp` and `faster-whisper`), in addition to the Python dependencies above.

### Connecting to Claude Desktop or Claude Code

Add an entry to your MCP client's server configuration (e.g. `claude_desktop_config.json` or `.mcp.json`):

```json
{
  "mcpServers": {
    "eduagent_mcp": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "/absolute/path/to/eduagent-os",
      "env": {
        "PYTHONPATH": ".",
        "GEMINI_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

Once connected, you can ask your client things like "process this lecture and give me the flashcards" and it will call `process_lecture` followed by `get_flashcards` automatically.

---

## Deploy to Hugging Face Spaces

The same container also serves as a live, interactive demo. `app.py` is a Gradio frontend (YouTube URL in, study notes / flashcards / evaluation tabs out) that the Docker image runs by default - `docker-compose.yml` is the only place that overrides it back to the one-shot batch pipeline for local use.

### 1. Create the Space

Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) with the **Docker** SDK selected. The YAML block at the top of this README already configures `sdk: docker` and `app_port: 7860`, so no manual Space settings changes are required beyond creating it.

### 2. Add your API key as a secret

In the Space's **Settings -> Variables and secrets**, add a new secret:

| Name | Value |
|---|---|
| `GEMINI_API_KEY` | your Google AI Studio API key |

Secrets are injected into the container as environment variables at runtime and are never visible in the Space's public files, matching how `GEMINI_API_KEY` is already read via `os.getenv` in `src/main.py`.

### 3. Push this repository to the Space

```bash
git remote add space https://huggingface.co/spaces/<your-username>/eduagent-os
git push space main
```

Hugging Face will build the existing `Dockerfile` (installing dependencies, baking the Whisper model, and installing Gradio) and launch `app.py` on port 7860. Build time is dominated by baking the `faster-whisper` base model; expect the first build to take several minutes.

### 4. Use the live demo

Once the Space is running, paste a YouTube URL into the textbox and click **Process Lecture**. Processing time depends on lecture length and the Space's hardware tier (CPU Basic is sufficient for short lectures but will be slower than a GPU-backed tier for longer ones).

Note: Hugging Face's persistent storage add-on has been discontinued, so `workspace/` artifacts are ephemeral and reset whenever the Space restarts - this is expected for a stateless demo.

---

## Security Model

Credentials are never stored in code or committed to version control. The `GEMINI_API_KEY` is passed into the container exclusively through Docker Compose's `environment` block, sourced from the local shell or a `.env` file that is explicitly excluded by `.gitignore`. The `workspace/` directory is also excluded to prevent accidental commits of transcribed audio content.

Beyond credential handling, the ingestion surface is hardened for public deployment: remote URLs are validated against a YouTube host whitelist before any network call (preventing the demo from being used as an SSRF proxy), local file paths are resolved with `os.path.realpath` and rejected unless they live inside the workspace directory (preventing arbitrary container file reads), and URLs without an extractable video ID are rejected rather than falling back to a shared cache filename. Transcript content is wrapped in explicit delimiters and every prompt instructs the model to treat it as data, mitigating prompt injection from malicious lecture audio. The container drops root privileges via a dedicated `appuser`, and the Gradio frontend gives each request its own temporary workspace so concurrent users can never read or overwrite each other's artifacts.

---

## Sample Output

See the `sample_output/` directory for real outputs generated by the pipeline from a Google AI Agents Intensive capstone overview lecture.

---

## CI Status

The GitHub Actions pipeline (`ci.yml`) runs on every push to `main`. It installs `flake8` and runs two passes: a hard-fail pass checking for syntax errors and undefined names (`E9, F63, F7, F82`), and a warning pass checking complexity and line length. The build fails if any syntax errors are detected.
