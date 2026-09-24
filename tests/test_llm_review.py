"""Tests for the LLM correction and label-review passes, with Ollama faked."""

import sys
import os
import json
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "conversation_analysis"))

for _mod in ["transformers", "vaderSentiment", "vaderSentiment.vaderSentiment"]:
    sys.modules.setdefault(_mod, MagicMock())

from llm_review import LLMReviewer  # noqa: E402
from raw_transcript_converter import RawTranscriptConverter  # noqa: E402


def _fake_ollama(monkeypatch, *answers):
    """Each call returns the next answer (a dict, serialised as the model's JSON reply)."""
    answers = iter(answers)
    calls = []

    def post(url, json, timeout):
        calls.append(json)
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"response": _dumps(next(answers))}
        return response

    monkeypatch.setattr("llm_review.requests.post", post)
    return calls


def _dumps(obj):
    return json.dumps(obj)


def _reviewer():
    return LLMReviewer("http://127.0.0.1:11434", "gemma3:1b")


class TestCorrections:

    def test_obvious_mishearing_is_fixed(self, monkeypatch):
        _fake_ollama(monkeypatch, {"lines": [{"id": 0, "child_reply": "Math"}]})
        result = _reviewer().correct_transcription([("What's your favourite subject?", "Mass")])
        assert result == ["Math"]

    def test_unchanged_reply_returns_none(self, monkeypatch):
        _fake_ollama(monkeypatch, {"lines": [{"id": 0, "child_reply": "Soccer and video games."}]})
        assert _reviewer().correct_transcription([("Fun?", "Soccer and video games.")]) == [None]

    def test_punctuation_or_case_only_is_not_a_correction(self, monkeypatch):
        _fake_ollama(monkeypatch, {"lines": [{"id": 0, "child_reply": "Yes."}]})
        assert _reviewer().correct_transcription([("Did you?", "yes")]) == [None]

    def test_rewrite_is_rejected(self, monkeypatch):
        _fake_ollama(monkeypatch, {"lines": [{"id": 0, "child_reply": "I really enjoy playing outside with friends"}]})
        assert _reviewer().correct_transcription([("Fun?", "Soccer.")]) == [None]

    def test_added_words_are_rejected(self, monkeypatch):
        _fake_ollama(monkeypatch, {"lines": [
            {"id": 0, "child_reply": "I like it because I am good at it and it is fun and easy for me"}
        ]})
        assert _reviewer().correct_transcription([("Why?", "I like it because I am good at it")]) == [None]

    def test_batches_of_ten(self, monkeypatch):
        replies = [("Q?", f"reply {i}") for i in range(23)]
        calls = _fake_ollama(monkeypatch, {"lines": []}, {"lines": []}, {"lines": []})
        _reviewer().correct_transcription(replies)
        assert len(calls) == 3

    def test_robot_question_given_as_context(self, monkeypatch):
        calls = _fake_ollama(monkeypatch, {"lines": []})
        _reviewer().correct_transcription([("What's your favourite subject?", "Mass")])
        assert "\"robot_question\": \"What's your favourite subject?\"" in calls[0]["prompt"]
        assert calls[0]["format"]["required"] == ["lines"]


class TestLabelReview:

    LINES = [
        {"speaker": "child", "text": "I love it!", "emotion": "neutral", "sentiment": "neutral"},
        {"speaker": "robot", "text": "Great!", "emotion": "joy", "sentiment": "positive"},
    ]

    def test_clear_mistake_is_changed(self, monkeypatch):
        _fake_ollama(monkeypatch, {"lines": [
            {"id": 0, "emotion": "joy", "sentiment": "positive"},
            {"id": 1, "emotion": "joy", "sentiment": "positive"},
        ]})
        assert _reviewer().review_labels(self.LINES) == [("joy", "positive"), ("joy", "positive")]

    def test_invalid_label_keeps_original(self, monkeypatch):
        _fake_ollama(monkeypatch, {"lines": [{"id": 0, "emotion": "happy", "sentiment": "great"}]})
        assert _reviewer().review_labels(self.LINES)[0] == ("neutral", "neutral")

    def test_missing_line_keeps_original(self, monkeypatch):
        _fake_ollama(monkeypatch, {"lines": []})
        assert _reviewer().review_labels(self.LINES) == [("neutral", "neutral"), ("joy", "positive")]

    def test_ollama_down_keeps_everything_and_stops_trying(self, monkeypatch):
        calls = []

        def refuse(*a, **k):
            calls.append(1)
            raise ConnectionError("refused")

        monkeypatch.setattr("llm_review.requests.post", refuse)
        r = _reviewer()
        lines = self.LINES * 15  # 30 lines = 3 batches
        assert r.review_labels(lines) == [(l["emotion"], l["sentiment"]) for l in lines]
        assert len(calls) == 1


class TestConverterIntegration:
    RAW = [
        {"time": "2026-05-06T18:00:00+00:00", "haru": "What's your favourite subject?", "user": "[START]"},
        {"time": "2026-05-06T18:00:05+00:00", "haru": "Math is fun!", "user": "Mass"},
    ]

    def _converter(self):
        c = RawTranscriptConverter(correct_transcription=True, review_labels=True)
        c._get_emotion = MagicMock(return_value=("neutral", 0.9))
        c._get_sentiment = MagicMock(return_value=("neutral", 0.1))
        c._assign_topics = lambda sentences, turns=None: ["School"] * len(sentences)
        c.vader = MagicMock()
        c.vader.polarity_scores.return_value = {"compound": 0.0}
        return c

    def test_correction_and_review_recorded(self, monkeypatch):
        _fake_ollama(
            monkeypatch,
            {"lines": [{"id": 0, "child_reply": "Math"}]},
            {"lines": [
                {"id": 0, "emotion": "neutral", "sentiment": "neutral"},
                {"id": 1, "emotion": "neutral", "sentiment": "neutral"},
                {"id": 2, "emotion": "joy", "sentiment": "positive"},
            ]},
        )
        out = self._converter().convert_json_transcript(self.RAW)

        child = out[1]
        assert child["sentence"] == "Math"
        assert child["original_sentence"] == "Mass"

        robot = out[2]
        assert robot["emotion_label"] == "joy"
        assert robot["emotion_label_before_review"] == "neutral"
        assert robot["sentiment_label_before_review"] == "neutral"
        assert "emotion_label_before_review" not in out[0]

    def test_emotion_runs_on_corrected_text(self, monkeypatch):
        _fake_ollama(monkeypatch, {"lines": [{"id": 0, "child_reply": "Math"}]}, {"lines": []})
        c = self._converter()
        c.convert_json_transcript(self.RAW)
        assert "Math" in [call.args[0] for call in c._get_emotion.call_args_list]

    def test_env_switches_turn_passes_off(self, monkeypatch):
        calls = _fake_ollama(monkeypatch)
        c = RawTranscriptConverter()  # conftest sets both env switches to 0
        c._get_emotion = MagicMock(return_value=("neutral", 0.9))
        c._get_sentiment = MagicMock(return_value=("neutral", 0.1))
        c._assign_topics = lambda sentences, turns=None: ["School"] * len(sentences)
        out = c.convert_json_transcript(self.RAW)
        assert calls == []
        assert out[1]["sentence"] == "Mass"


def test_prompt_echo_is_rejected():
    r = _reviewer()
    original = "uh basically salad with tomatoes really sliced uh really small sliced tomatoes"
    echoed = 'Robot asked: "Can you tell me more?" ' + original
    assert r._accept_correction(original, echoed, "Can you tell me more?") is None


def test_two_word_fix_in_short_reply_accepted():
    r = _reviewer()
    assert r._accept_correction("Have you game character?", "A video game character?") == \
        "A video game character?"


def test_many_changed_words_rejected():
    r = _reviewer()
    assert r._accept_correction("I went to the park with dad", "I visited a garden with mum") is None


class TestRealExamples:
    """Corrections seen from gemma3:1b on real transcripts: good ones kept, bad ones rejected."""

    GOOD = [("Mass", "Math"), ("Waken", "Raiken"), ("Wake in.", "Waken"),
            ("Have you game character?", "A video game character?")]
    BAD = [("Yes.", "Yesterday."), ("Ah, being smart.", "Ah, being famous someday!"),
           ("Um, probably I'd want to find how to cure cancer",
            "Probably I'd want to find how to cure cancer"),
           ("My favorite food is pierogies.", "favorite food is pierogies."),
           ("Probably telekinesis.", "Not really."), ("Sushi", "Raiken"), ("Soccer", "Yes.")]

    def test_good_fixes_accepted(self):
        for original, corrected in self.GOOD:
            assert _reviewer()._accept_correction(original, corrected) == corrected, original

    def test_bad_fixes_rejected(self):
        for original, corrected in self.BAD:
            assert _reviewer()._accept_correction(original, corrected) is None, original


class TestVaderVeto:

    def _converter(self, compound):
        c = RawTranscriptConverter(review_labels=True)
        c.vader = MagicMock()
        c.vader.polarity_scores.return_value = {"compound": compound}
        c.reviewer = MagicMock()
        return c

    def _line(self, emotion="joy", sentiment="positive"):
        return {"turn": "robot", "sentence": "Always have fun!", "emotion_label": emotion,
                "sentiment_label": sentiment}

    def test_negative_label_on_positive_text_vetoed(self):
        c = self._converter(0.8)
        c.reviewer.review_labels.return_value = [("anger", "negative")]
        line = self._line()
        c._review_conversation_labels([line])
        assert (line["emotion_label"], line["sentiment_label"]) == ("joy", "positive")
        assert "emotion_label_before_review" not in line

    def test_label_agreeing_with_vader_stands(self):
        c = self._converter(0.8)  # clearly positive, and the label is joy/positive
        c.reviewer.review_labels.return_value = [("neutral", "neutral")]
        line = self._line()
        c._review_conversation_labels([line])
        assert (line["emotion_label"], line["sentiment_label"]) == ("joy", "positive")

    def test_disagreement_settled_by_review(self):
        c = self._converter(0.8)  # clearly positive, but the emotion model said sadness
        c.reviewer.review_labels.return_value = [("joy", "positive")]
        line = self._line("sadness", "negative")
        c._review_conversation_labels([line])
        assert (line["emotion_label"], line["sentiment_label"]) == ("joy", "positive")
        assert line["emotion_label_before_review"] == "sadness"

    def test_any_change_allowed_on_mixed_text(self):
        c = self._converter(0.0)
        c.reviewer.review_labels.return_value = [("sadness", "negative")]
        line = self._line()
        c._review_conversation_labels([line])
        assert line["emotion_label"] == "sadness"


    def test_short_line_not_reviewed(self):
        c = self._converter(0.0)
        c.reviewer.review_labels.return_value = [("sadness", "negative")]
        line = dict(self._line("neutral", "neutral"), sentence="Aria")
        c._review_conversation_labels([line])
        assert (line["emotion_label"], line["sentiment_label"]) == ("neutral", "neutral")
