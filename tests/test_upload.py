"""
Tests for the browser upload handler in app.py.

The processing pipeline is mocked, so these check routing and messages only:
raw Haru JSON is processed, processed JSON is shown as-is, anything else is rejected.
"""

import sys
import os
import json
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "conversation_analysis"))

for _mod in ["transformers", "vaderSentiment", "vaderSentiment.vaderSentiment"]:
    sys.modules.setdefault(_mod, MagicMock())

import app as app_module  # noqa: E402
import process_uploaded_transcript as pipeline  # noqa: E402


RAW = [
    {"time": "2026-05-06T18:00:00+00:00", "haru": "Hi, my name is Haru!"},
    {"time": "2026-05-06T18:00:02+00:00", "haru": "What is your name?", "user": "Bob."},
]

PROCESSED = [{
    "idx": 0, "sentence": "Hi", "turn": "robot", "slots": [],
    "emotion_label": "joy", "sentiment_label": "positive",
}]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "TRANSCRIPT_FILE_PATH", str(tmp_path / "transcript.json"))
    monkeypatch.setattr(app_module, "DIAGRAM_FILE_PATH", str(tmp_path / "diagram.json"))
    monkeypatch.setattr(app_module, "generate_diagram_from_file", MagicMock())
    monkeypatch.setattr(app_module, "_ollama_reachable", lambda host: True)
    monkeypatch.setattr(pipeline, "RAW_UPLOADS_DIR", str(tmp_path / "raw_uploads"))

    converter = MagicMock(topic_method="model", ollama_host="http://localhost:11434")
    monkeypatch.setattr(pipeline, "_converter", converter)
    monkeypatch.setattr(pipeline, "process_haru_json", MagicMock(return_value=([], "P1_Bob.json")))

    return app_module.socketio.test_client(app_module.app)


def _events(client):
    return [(m["name"], m["args"]) for m in client.get_received()]


def test_raw_haru_json_is_processed(client, tmp_path):
    client.emit("upload_file", json.dumps(RAW).encode(), "P1_Bob.json")
    events = _events(client)

    pipeline.process_haru_json.assert_called_once_with(RAW, "P1_Bob.json")
    assert ("upload-success", []) in events
    assert ("upload-progress", ["Saved as P1_Bob.json"]) in events
    assert (tmp_path / "raw_uploads" / "processed_P1_Bob.json").exists()


def test_processed_json_skips_pipeline(client):
    client.emit("upload_file", json.dumps(PROCESSED).encode(), "P1_Bob.json")
    events = _events(client)

    pipeline.process_haru_json.assert_not_called()
    assert ("upload-success", []) in events


def test_unrecognised_json_is_rejected(client):
    client.emit("upload_file", b'[{"foo": 1}]', "x.json")
    names = [name for name, _ in _events(client)]

    pipeline.process_haru_json.assert_not_called()
    assert names == ["upload-error"]


def test_non_json_extension_is_rejected(client):
    client.emit("upload_file", b"robot: hi", "x.yml")
    assert [name for name, _ in _events(client)] == ["upload-error"]


def test_warns_when_topic_model_unreachable(client, monkeypatch):
    monkeypatch.setattr(app_module, "_ollama_reachable", lambda host: False)
    client.emit("upload_file", json.dumps(RAW).encode(), "P1_Bob.json")
    progress = [args[0] for name, args in _events(client) if name == "upload-progress"]

    assert any(msg.startswith("Warning: topic model not reachable") for msg in progress)


def test_path_in_filename_is_stripped(client, tmp_path):
    client.emit("upload_file", json.dumps(RAW).encode(), "../../evil/P1_Bob.json")
    _events(client)

    pipeline.process_haru_json.assert_called_once_with(RAW, "P1_Bob.json")
    assert (tmp_path / "raw_uploads" / "processed_P1_Bob.json").exists()


def test_pipeline_error_reported_to_browser(client):
    pipeline.process_haru_json.side_effect = RuntimeError("boom")
    client.emit("upload_file", json.dumps(RAW).encode(), "P1_Bob.json")

    assert ("upload-error", ["boom"]) in _events(client)
    assert app_module.processing_lock.acquire(blocking=False)
    app_module.processing_lock.release()


def test_processed_prefix_stripped_from_filename(client, tmp_path):
    client.emit("upload_file", json.dumps(RAW).encode(), "processed_P3_Surena.json")
    _events(client)

    pipeline.process_haru_json.assert_called_once_with(RAW, "P3_Surena.json")
    assert (tmp_path / "raw_uploads" / "processed_P3_Surena.json").exists()


def test_start_marker_hidden_in_processed_upload(client, tmp_path):
    processed = [dict(PROCESSED[0], idx=0, sentence="[START]", turn="user", last_interaction=False),
                 dict(PROCESSED[0], idx=1, sentence="Hi", last_interaction=False),
                 dict(PROCESSED[0], idx=2, sentence="Hello", turn="user", last_interaction=True)]
    client.emit("upload_file", json.dumps(processed).encode(), "P1_Bob.json")
    assert ("upload-success", []) in _events(client)

    shown = json.loads((tmp_path / "transcript.json").read_text(encoding="utf8"))
    assert [u["sentence"] for u in shown] == ["Hi", "Hello"]
    assert [u["idx"] for u in shown] == [0, 1]
    assert shown[-1]["last_interaction"] is True


def test_file_without_extension_is_processed(client):
    client.emit("upload_file", json.dumps(RAW).encode(), "P16-Ryken")
    events = _events(client)

    pipeline.process_haru_json.assert_called_once_with(RAW, "P16-Ryken")
    assert ("upload-success", []) in events


def test_non_json_content_rejected_with_message(client):
    client.emit("upload_file", b"just some notes", "notes.txt")
    events = _events(client)
    assert events == [("upload-error", ['"notes.txt" is not a JSON transcript.'])]


def test_empty_file_gets_an_answer(client):
    client.emit("upload_file", b"", "Cody-p8")
    assert _events(client) == [("upload-error", ['"Cody-p8" is empty.'])]
