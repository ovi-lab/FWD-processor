# Haru Conversation Visualizer

A pipeline for processing Haru robot transcripts and visualising the conversations.

The system has two separate commands — run the processing script once after a session to produce the output JSON, then run the diagram server to view it. The diagram server never loads ML models or Ollama.

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
```

Every time you open a new terminal:

```bash
# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Ollama

The processing script uses a local Ollama model for topic labelling. Install Ollama from https://ollama.com, then pull the model:

```bash
ollama pull llama3.2:1b
```

Ollama only needs to be running when you run the processing script. If it is not running, topic labels will fall back to keyword extraction automatically — processing will still complete.

---

## Command 1 — Process transcripts

Run this after a session to convert raw transcripts into processed JSON.

```bash
python conversation_analysis/process_uploaded_transcript.py
```

### Supported input formats

**Haru JSON** (`.json`) — the native output format from the robot:

```json
[
    {"time": "2026-05-06T18:00:00+00:00", "haru": "Hi, my name is Haru!"},
    {"time": "2026-05-06T18:00:02+00:00", "haru": "What is your name?", "user": "My name is Bob."},
    {"time": "2026-05-06T18:00:10+00:00", "haru": "Nice to meet you, Bob!"}
]
```

Name the file `P{id}_{name}.json` to set the participant ID and child name (e.g. `P1_Bob.json`). If the filename does not match that pattern, the script reads the child's name from Haru's closing line ("Thank you, Bob!") and falls back to the filename otherwise.

**Labeled TXT** (`.txt`):

```
{P1}
{Bob}
{'robot': 'Hello! What is your name?'},
{'user': 'Hi, my name is Bob.'},
{'robot': 'Nice to meet you, Bob!'},
```

Requirements:
- Line 1: participant ID as `{P1}`, `{P2}`, etc.
- Line 2: child's name as `{Bob}`, `{Alice}`, etc.
- Speaker labels must be one of: `robot`, `haru`, `user`, `child`
- Each line wrapped in `{}` with single quotes, ending with a comma

### Steps

1. Place raw transcript files into `data/raw_uploads/`
2. Start Ollama
3. Run the script

For each unprocessed file the script will:
1. Detect the file format
2. Extract participant ID and child name
3. Run emotion, sentiment, and topic analysis on every utterance
4. Save the processed JSON to `data/named_outputs/P1_Bob.json`
5. Update `data/transcript_data/transcript-log.json`
6. Rename the original file to `processed_P1_Bob.json` so it is not picked up again

Files already prefixed with `processed_` are skipped. Safe to re-run — only new files are processed.

### Output

| File | Description |
|------|-------------|
| `data/named_outputs/P1_Bob.json` | One file per session, named by participant |
| `data/transcript_data/transcript-log.json` | Latest transcript (read by the diagram server) |
| `data/diagram_data/diagram-log.json` | Latest diagram data |

---

## Command 2 — Run the diagram server

Run this to view and interact with processed transcripts. Does not load any ML models or require Ollama.

```bash
python conversation_analysis/app.py
```

This starts the backend on port **6400**. You must also run the frontend visualizer (see below) — the diagram is not visible from the backend alone.

---

## Command 3 — Run the frontend visualizer

The frontend lives in the **Haru-Chat-Visualizer** repository. Set it up once:

```bash
npm install
```

Then start the dev server:

```bash
npm run dev
```

This starts the visualizer on port **8080**. Open `http://localhost:8080` in your browser.

> Both the backend (`app.py`) and the frontend (`npm run dev`) must be running at the same time to use the visualizer.

### Using the interface

- **Load a transcript** — upload a processed JSON file from `data/named_outputs/` via the browser interface
- **View the diagram** — the visualisation updates automatically after upload completes
- **Live feed** — toggle the live feed switch to stream data from an active ROS session
- **Download** — export the current transcript as JSON from the interface

> The diagram server only accepts pre-processed JSON. Run Command 1 first to produce the processed file, then upload it here.

---

## Typical workflow

```
Place transcript in data/raw_uploads/
        ↓
python conversation_analysis/process_uploaded_transcript.py
        ↓
data/named_outputs/P1_Bob.json produced
        ↓
python conversation_analysis/app.py   (terminal 1)
npm run dev                           (terminal 2)
        ↓
Open http://localhost:8080
        ↓
Upload P1_Bob.json via browser → diagram loads automatically
```
