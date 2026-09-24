# Haru Conversation Processor

The processing backend for Haru robot transcripts. It turns a raw transcript into processed JSON (emotion, sentiment, topics) and serves it to the web page.

- **`app.py`** — the backend server. The web page connects to it, and raw transcripts uploaded in the browser are processed here.
- **`process_uploaded_transcript.py`** — a batch script for processing a folder of transcripts without the browser.

To **see** the conversations you also need the web page, **Haru-Chat-Visualizer** (https://github.com/ovi-lab/Haru-Chat-Visualizer). Set up this repo first, then follow the visualizer's README.

---

## Setup

**Requirements** (Windows or Linux):

- Python **3.11+** (3.12 recommended — note Ubuntu 22.04 ships 3.10, which is too old for the pinned packages)
- [Ollama](https://ollama.com)
- An NVIDIA GPU with ~10GB of memory is recommended for `gemma4:e4b`; everything also runs on the CPU, slowly

```bash
git clone https://github.com/juliapetrie/Julia-CDV-processor.git Haru-Chat-Processor
cd Haru-Chat-Processor
git checkout Julia-Updated-Processor

python -m venv venv
# activate: venv\Scripts\activate (Windows) or source venv/bin/activate (Linux)
pip install -r requirements.txt pytest

ollama pull gemma4:e4b
```

- `gemma4:e4b` (9.6GB) is the default topic model, so no configuration is needed. On a smaller GPU, pull `gemma4:e2b` (7.2GB) or `gemma3:1b` (815MB) and set `HARU_TOPIC_MODEL` accordingly (see *Settings*).
- The emotion and embedding models (~400MB) download from Hugging Face on the first run.
- **`data/` is not in git** — it holds children's transcripts. Copy `data/named_outputs/` and `data/raw_uploads/` from the old machine over a secure channel (not a personal cloud drive or GitHub).

**Check it works:** `python -m pytest tests -q` should pass, and `python conversation_analysis/app.py` should print `Topic model gemma4:e4b loaded in Ollama.` A different model name means `HARU_TOPIC_MODEL` is set; `Model warm-up skipped` means Ollama isn't running or the model isn't pulled.

Then set up the web page from the **Haru-Chat-Visualizer** README.

---

## Running the backend (`app.py`)

```bash
python conversation_analysis/app.py
```

Serves the web page on port **6400**; keep it running while the page is in use. At startup it loads the processing models (and the Gemma model in Ollama) in the background so the first upload is quick.

- **Restart it after changing any Python code** — it only reads the code at startup.
- `HARU_SKIP_WARMUP=1` skips model loading at startup (useful on a view-only machine).
- The `unauthenticated requests to the HF Hub` warning at startup is harmless. `HF_HUB_OFFLINE=1` silences it and starts faster, but unset it before switching to a model that isn't downloaded yet.

### What happens to uploads

The web page sends each uploaded file here. The backend decides by the file's **contents** (the extension doesn't matter):

| Uploaded file | What the backend does |
|---|---|
| **Raw Haru transcript** (`[{"time", "haru", "user"}, ...]`) | Processes it (see *How processing works*; about 15–25 seconds with `gemma3:1b`), saves `data/named_outputs/<ID>_<Name>.json`, keeps a copy of the raw file as `data/raw_uploads/processed_<filename>`, and sends progress messages to the page |
| **Processed transcript** (from `data/named_outputs/`) | Shows it as saved, in about a second. Nothing is re-run |
| Anything else (notes, empty files) | Rejected with a message |

Uploading the raw file again re-runs everything and overwrites the saved result; topic names may come out slightly differently. To re-open a transcript exactly as it was, upload its file from `data/named_outputs/`.

---

## Batch processing (`process_uploaded_transcript.py`)

Processes every new transcript in a folder, without the browser.

```bash
python conversation_analysis/process_uploaded_transcript.py
```

It picks up `.json` and `.txt` files in `data/raw_uploads/` (browser uploads also accept files without an extension). For each file it extracts the participant ID and child name, processes the transcript (see *How processing works*), saves `data/named_outputs/<ID>_<Name>.json`, and renames the original to `processed_<filename>` so it isn't picked up again. Files already prefixed with `processed_` are skipped, so it's safe to re-run.

### Supported input formats

**Haru JSON** — the native output format from the robot:

```json
[
    {"time": "2026-05-06T18:00:00+00:00", "haru": "Hi, my name is Haru!"},
    {"time": "2026-05-06T18:00:02+00:00", "haru": "What is your name?", "user": "My name is Bob."},
    {"time": "2026-05-06T18:00:10+00:00", "haru": "Nice to meet you, Bob!"}
]
```

The participant ID and child name come from the filename (for browser uploads too). All of these work:

| Filename | ID / name |
|---|---|
| `P1_Bob.json`, `P1-Bob.json`, `P1-Bob` (no extension) | P1 / Bob |
| `16_Raiken.json` (no `P`) | P16 / Raiken |
| `P19.1-Alex` (split session) | P19.1 / Alex |
| `Lucy5.1` (name first) | P5.1 / Lucy |
| `processed_P3_Surena.json` (re-uploaded from `data/raw_uploads/`) | P3 / Surena |

If none match, the ID is `P0` and the name is read from Haru's closing line ("Thank you, Bob!"), falling back to the filename.

Whole-line system commands such as `"user": "[START]"` are removed before processing.

**Labeled TXT** (`.txt`, batch script only):

```
{P1}
{Bob}
{'robot': 'Hello! What is your name?'},
{'user': 'Hi, my name is Bob.'},
{'robot': 'Nice to meet you, Bob!'},
```

- Line 1: participant ID as `{P1}`, `{P2}`, etc.
- Line 2: child's name as `{Bob}`, `{Alice}`, etc.
- Speaker labels must be one of: `robot`, `haru`, `user`, `child`
- Each line wrapped in `{}` with single quotes, ending with a comma

### Output files

| File | Description |
|---|---|
| `data/named_outputs/<ID>_<Name>.json` | One file per session. Re-processing the same participant overwrites it |
| `data/raw_uploads/processed_<filename>` | The raw transcript, marked as processed |
| `data/transcript_data/transcript-log.json` | The transcript currently shown in the web page (overwritten by every upload) |
| `data/diagram_data/diagram-log.json` | The diagram currently shown in the web page (overwritten by every upload) |

---

## How processing works

Processing (browser uploads and the batch script) uses the local Ollama model to name topics, lightly fix transcription mistakes and check emotion labels. Ollama must be running while transcripts are processed. If it isn't, processing still completes: topics fall back to keyword labels and the transcription and label checks are skipped (browser uploads show a warning).

### How topics are found

1. Lines are grouped into exchanges (a Haru line plus the child's reply) and embedded with a local sentence-transformers model.
2. The transcript is split at the 8 biggest drops in similarity between neighbouring exchanges, giving 9 topics for a full session (`max_topics`, fewer for short transcripts). Splits are never adjacent, so every topic has at least two exchanges.
3. At each split, the child is credited with changing the topic if their last reply is clearly closer to the next topic's question than to the one they answered; otherwise Haru is. The diagram draws the topic change from that speaker.
4. Each topic is sent to the local Ollama model, which names it in 1–2 words. Names are never reused, so the diagram flows clockwise without looping back.

### Emotion and sentiment

Each line gets an emotion from `j-hartmann/emotion-english-distilroberta-base` and a sentiment from that emotion plus VADER. The topic model then runs two light checks (skipped if Ollama isn't running):

1. **Transcription fixes** (before any analysis): obvious speech-to-text mistakes in the child's lines are corrected, e.g. "Mass" -> "Math" after "What's your favourite subject?". Only swaps of similar-sounding words are accepted; nothing is added, deleted or reworded. The original is kept in `original_sentence`. Haru's lines are never changed.
2. **Label check** (after the models): the topic model is a third opinion. Where the emotion model and VADER agree on the tone, their label stands; the topic model mainly settles lines where they disagree or the tone is mixed, and can never give a negative label to clearly positive text (or the reverse). Lines under 3 words are left alone. Changed labels keep the originals in `emotion_label_before_review` / `sentiment_label_before_review`.

These add roughly 10–20 seconds per transcript with `gemma3:1b`.

### Settings

All optional, set as environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `HARU_TOPIC_MODEL` | `gemma4:e4b` | Ollama model that names topics and runs the checks |
| `HARU_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model that detects topic shifts |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server (e.g. a GPU machine on the network) |
| `HARU_MAX_LABEL_CHARS` | `20` | Topic labels longer than this are cut to their first word |
| `HARU_CORRECT_TRANSCRIPTS` | `1` | `0` turns off light speech-to-text fixes in the child's lines |
| `HARU_REVIEW_LABELS` | `1` | `0` turns off the topic model's check of emotion/sentiment labels |
| `HARU_SKIP_WARMUP` | unset | `1` stops `app.py` loading models at startup |

The same settings can be passed as constructor arguments (`RawTranscriptConverter(topic_model=..., embedding_model=..., ollama_host=..., max_label_chars=..., max_topics=..., correct_transcription=..., review_labels=...)`). To use Haru's fixed-script trigger phrases instead of a model for topics, pass `topic_method="script"`.

---

# JSONL to JSON Converter

Converts `.jsonl` files (one JSON object per line) into a single JSON array.

## Requirements

Python 3 (no extra packages needed).

## Usage

From the `JSON Converter` folder:

```powershell
python jsonl_to_json.py filename
```

Output is saved as `filename.json` in the same folder.

### Options

```powershell
# custom output path
python jsonl_to_json.py P1-Alice P1-Alice-out.json

# custom indent (default is 2)
python jsonl_to_json.py P1-Alice --indent=4
```

### Convert multiple files at once

```powershell
foreach ($f in "P1-Alice","P2-Calla","P3-Surena","P4-Cash") { python jsonl_to_json.py $f }
```

## Notes

- Files do not need a `.jsonl` extension
- Empty lines are skipped
- If a line fails to parse, the script will tell you which line number and stop
