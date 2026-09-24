import pytest


@pytest.fixture(autouse=True)
def no_llm_review_by_default(monkeypatch):
    """Tests never call a real Ollama; LLM review tests turn these back on explicitly."""
    monkeypatch.setenv("HARU_CORRECT_TRANSCRIPTS", "0")
    monkeypatch.setenv("HARU_REVIEW_LABELS", "0")
