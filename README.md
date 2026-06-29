# EduAgent-OS 🚀

An open-source, local-first multi-agent engine built on the **Google Antigravity 2.0 SDK**. It automatically ingests remote lecture URLs or local media files, executing on-device transcription and context compaction to generate structured Markdown notes and Anki-ready study flashcards.

## 🌟 Key Features

* **Local-First Privacy:** All media processing and transcriptions happen inside an isolated container boundary via `faster-whisper`. No data is shared or used for training.
* **Multi-Agent Orchestration:** Uses the Antigravity ADK to spawn specialized parallel sub-agents (*Synthesis Specialist* and *Educational Taxonomist*).
* **Context Compaction Layer:** Automatically handles long, multi-hour lectures without context degradation or token overflow bugs.
* **Deterministic Evaluation:** Features an automated self-correction guard layer using structured Pydantic validation schemas to verify facts and eliminate hallucinations.

## 🛠️ Architecture

* **Orchestration:** Google Antigravity SDK (`antigravity-preview-05-2026`)
* **Inference Model:** Gemini 3.5 Flash (via Google AI Studio)
* **Acoustic Layer:** `faster-whisper` (C++ Implementation, CPU `int8` optimized)
* **Sourcing Engine:** `yt-dlp` + `ffmpeg`

## 🚀 Quick Start (Production Setup)

### 1. Prerequisites
Ensure you have [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed on your host machine.

### 2. Clone and Configure
```bash
git clone [https://github.com/yourusername/eduagent-os.git](https://github.com/yourusername/eduagent-os.git)
cd eduagent-os
