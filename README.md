# 🩺 Doc Assistant

AI-powered post-visit assistant that records your doctor's appointment, translates it into plain language, creates an action checklist, and schedules follow-up appointments via MyChart.

## Features

- 🎙️ **Record or upload** appointment audio in the browser
- 🗣️ **Transcription** via Groq Whisper (free)
- 🤖 **Plain-language summary** via Claude (Anthropic)
- ✅ **Action item checklist** — medications, tests, lifestyle changes
- 💊 **Medication cards** with pickup reminders
- 📅 **MyChart scheduling** via Epic FHIR R4 + SMART on FHIR OAuth2

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/doc-assistant.git
cd doc-assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add API keys
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

```
GROQ_API_KEY=your-groq-key        # free at console.groq.com
ANTHROPIC_API_KEY=your-claude-key # console.anthropic.com
EPIC_CLIENT_ID=your-epic-client-id # open.epic.com (optional)
```

### 4. Run the app
```bash
python app.py
```

Open **http://localhost:7860** in your browser.

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Gradio |
| Backend | FastAPI + Python |
| Transcription | Groq Whisper (whisper-large-v3) |
| AI Analysis | Anthropic Claude (claude-sonnet) |
| Scheduling | Epic FHIR R4 + SMART on FHIR |

## Project Structure

```
doc-assistant/
├── app.py              # Main Gradio + FastAPI app
├── requirements.txt
├── .env.example
├── frontend/
│   └── index.html      # Original web UI (reference)
├── backend/
│   └── main.py         # Original FastAPI backend (reference)
└── project_plan.md     # Course project plan
```
