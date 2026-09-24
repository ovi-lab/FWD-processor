"""
Tests for the Haru JSON transcript parsing pipeline.

Heavy ML dependencies (emotion model, embedding model, Ollama) are mocked so
the suite runs without a GPU or a running Ollama server.
"""

import sys
import os
import pytest
from unittest.mock import MagicMock

# Make conversation_analysis importable from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "conversation_analysis"))

# Stub out heavy ML packages before raw_transcript_converter is imported for
# the first time, so the module loads without GPU/model-download requirements.
_MOCK_MODULES = [
    "transformers",
    "vaderSentiment",
    "vaderSentiment.vaderSentiment",
]
for _mod in _MOCK_MODULES:
    sys.modules.setdefault(_mod, MagicMock())

from raw_transcript_converter import RawTranscriptConverter  # noqa: E402
import process_uploaded_transcript  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def converter():
    """
    Returns a RawTranscriptConverter with all external dependencies mocked.
    _get_emotion, _get_sentiment, and _assign_topics are replaced with simple
    stubs so tests focus on parsing logic, not ML correctness.
    """
    c = RawTranscriptConverter()
    c._get_emotion = MagicMock(return_value=("joy", 0.9))
    c._get_sentiment = MagicMock(return_value=("positive", 0.5))
    c._assign_topics = lambda sentences, turns=None: ["Test Topic"] * len(sentences)
    return c


SAMPLE_JSON = [
    {"time": "2026-05-06T18:00:00+00:00", "haru": "Hi, my name is Haru!"},
    {"time": "2026-05-06T18:00:02+00:00", "haru": "What is your name?", "user": "My name is Bob."},
    {"time": "2026-05-06T18:00:10+00:00", "haru": "Nice to meet you, Bob!"},
    {"time": "2026-05-06T18:00:12+00:00", "haru": "What is your favourite food?", "user": "Pizza."},
    {"time": "2026-05-06T18:00:20+00:00", "haru": "Thank you, Bob! This has been fun!"},
]


# ---------------------------------------------------------------------------
# convert_json_transcript — utterance count and structure
# ---------------------------------------------------------------------------

class TestConvertJsonTranscript:

    def test_haru_only_entry_produces_one_utterance(self, converter):
        data = [{"time": "2026-05-06T18:00:00+00:00", "haru": "Hello there!"}]
        result = converter.convert_json_transcript(data)
        assert len(result) == 1

    def test_haru_and_user_entry_produces_two_utterances(self, converter):
        data = [{"time": "2026-05-06T18:00:00+00:00", "haru": "What is your name?", "user": "Bob."}]
        result = converter.convert_json_transcript(data)
        assert len(result) == 2

    def test_mixed_entries_correct_total_count(self, converter):
        # SAMPLE_JSON has 3 haru-only and 2 haru+user entries → 3 + 2*2 = 7
        result = converter.convert_json_transcript(SAMPLE_JSON)
        assert len(result) == 7

    def test_haru_utterances_have_robot_turn(self, converter):
        result = converter.convert_json_transcript(SAMPLE_JSON)
        haru_turns = [u["turn"] for u in result if "Haru" in u["sentence"] or "meet" in u["sentence"] or "Hi" in u["sentence"]]
        assert all(t == "robot" for t in haru_turns)

    def test_user_utterances_have_user_turn(self, converter):
        result = converter.convert_json_transcript(SAMPLE_JSON)
        user_turns = [u["turn"] for u in result if u["sentence"] in ("My name is Bob.", "Pizza.")]
        assert all(t == "user" for t in user_turns)
        assert len(user_turns) == 2

    def test_sequential_idx(self, converter):
        result = converter.convert_json_transcript(SAMPLE_JSON)
        assert [u["idx"] for u in result] == list(range(len(result)))

    def test_only_last_entry_has_last_interaction_true(self, converter):
        result = converter.convert_json_transcript(SAMPLE_JSON)
        assert result[-1]["last_interaction"] is True
        assert all(not u["last_interaction"] for u in result[:-1])

    def test_required_output_fields_present(self, converter):
        required = {
            "idx", "sentence", "emotion_label", "emotion_score",
            "sentiment_label", "sentiment_score", "turn", "last_interaction",
            "highlighted", "intent", "intent_category", "timestamp", "topic",
            "type", "slots",
        }
        result = converter.convert_json_transcript(SAMPLE_JSON)
        for utterance in result:
            assert required.issubset(utterance.keys()), f"Missing keys in: {utterance}"

    def test_default_field_values(self, converter):
        result = converter.convert_json_transcript(SAMPLE_JSON)
        for u in result:
            assert u["highlighted"] is False
            assert u["intent"] == "default-intent"
            assert u["intent_category"] == "default"
            assert u["type"] == "text"
            assert u["slots"] == []

    def test_haru_only_timestamp_matches_entry_time(self, converter):
        # Opening greeting has no user turn — Haru gets the exact entry timestamp, no +1s offset.
        data = [{"time": "2026-05-06T18:00:00+00:00", "haru": "Hello!"}]
        result = converter.convert_json_transcript(data)
        assert result[0]["timestamp"] == "2026-05-06 18:00:00"

    def test_user_timestamp_is_one_second_after_haru(self, converter):
        data = [{"time": "2026-05-06T18:00:00+00:00", "haru": "Hi!", "user": "Hey!"}]
        result = converter.convert_json_transcript(data)
        assert result[0]["timestamp"] == "2026-05-06 18:00:00"
        assert result[1]["timestamp"] == "2026-05-06 18:00:01"

    def test_invalid_timestamp_falls_back_without_crashing(self, converter):
        data = [{"time": "not-a-date", "haru": "Hello!"}]
        result = converter.convert_json_transcript(data)
        assert len(result) == 1
        assert "timestamp" in result[0]

    def test_empty_haru_string_is_skipped(self, converter):
        data = [{"time": "2026-05-06T18:00:00+00:00", "haru": "", "user": "Hi!"}]
        result = converter.convert_json_transcript(data)
        assert len(result) == 1
        assert result[0]["turn"] == "user"

    def test_missing_user_key_produces_one_utterance(self, converter):
        data = [{"time": "2026-05-06T18:00:00+00:00", "haru": "Hello!"}]
        result = converter.convert_json_transcript(data)
        assert len(result) == 1

    def test_raises_on_empty_data(self, converter):
        with pytest.raises(ValueError, match="No transcript entries"):
            converter.convert_json_transcript([])

    def test_raises_when_all_haru_empty(self, converter):
        data = [{"time": "2026-05-06T18:00:00+00:00", "haru": ""}]
        with pytest.raises(ValueError, match="No transcript entries"):
            converter.convert_json_transcript(data)

    def test_emotion_and_sentiment_values_stored(self, converter):
        result = converter.convert_json_transcript(SAMPLE_JSON)
        for u in result:
            assert u["emotion_label"] == "joy"
            assert u["emotion_score"] == 0.9
            assert u["sentiment_label"] == "positive"
            assert u["sentiment_score"] == 0.5

    def test_topic_assigned_to_every_utterance(self, converter):
        result = converter.convert_json_transcript(SAMPLE_JSON)
        assert all(u["topic"] == "Test Topic" for u in result)

    def test_sentence_text_preserved(self, converter):
        # User speaks first, then Haru responds — so user is index 0, haru is index 1.
        data = [{"time": "2026-05-06T18:00:00+00:00", "haru": "Hello!", "user": "Hi there!"}]
        result = converter.convert_json_transcript(data)
        assert result[0]["sentence"] == "Hi there!"
        assert result[1]["sentence"] == "Hello!"


# ---------------------------------------------------------------------------
# extract_metadata_from_json
# ---------------------------------------------------------------------------

class TestExtractMetadataFromJson:

    def test_extracts_name_from_closing_statement(self, converter):
        data = [
            {"time": "2026-05-06T18:00:00+00:00", "haru": "Hello!"},
            {"time": "2026-05-06T18:00:02+00:00", "haru": "Thank you, Bob! This has been fun!"},
        ]
        _, child_name = converter.extract_metadata_from_json(data)
        assert child_name == "Bob"

    def test_participant_id_is_always_none(self, converter):
        participant_id, _ = converter.extract_metadata_from_json(SAMPLE_JSON)
        assert participant_id is None

    def test_returns_none_when_no_pattern_match(self, converter):
        data = [{"time": "2026-05-06T18:00:00+00:00", "haru": "Goodbye!"}]
        _, child_name = converter.extract_metadata_from_json(data)
        assert child_name is None

    def test_works_with_fewer_than_five_entries(self, converter):
        data = [
            {"time": "2026-05-06T18:00:00+00:00", "haru": "Hello!"},
            {"time": "2026-05-06T18:00:02+00:00", "haru": "Thank you, Alice! Bye!"},
        ]
        _, child_name = converter.extract_metadata_from_json(data)
        assert child_name == "Alice"

    def test_only_searches_last_five_entries(self, converter):
        # Name appears only in entry 0, which is more than 5 from the end
        data = (
            [{"time": "2026-05-06T18:00:00+00:00", "haru": "Thank you, Bob!"}]
            + [{"time": "2026-05-06T18:00:02+00:00", "haru": "Okay."} for _ in range(6)]
        )
        _, child_name = converter.extract_metadata_from_json(data)
        assert child_name is None

    def test_uses_full_json_when_five_or_fewer_entries(self, converter):
        data = [{"time": "2026-05-06T18:00:00+00:00", "haru": "Thank you, Charlie!"}]
        _, child_name = converter.extract_metadata_from_json(data)
        assert child_name == "Charlie"


# ---------------------------------------------------------------------------
# _is_haru_json_format (via process_uploaded_transcript module)
# ---------------------------------------------------------------------------

class TestIsHaruJsonFormat:

    @pytest.fixture(autouse=True)
    def import_helper(self):
        # module already imported at top of file — no local import needed
        self.fn = process_uploaded_transcript._is_haru_json_format

    def test_detects_new_haru_format(self):
        data = [{"time": "2026-05-06T18:00:00+00:00", "haru": "Hello!"}]
        assert self.fn(data) is True

    def test_rejects_old_processed_format(self):
        data = [{"idx": 0, "sentence": "Hello!", "turn": "robot"}]
        assert self.fn(data) is False

    def test_rejects_empty_list(self):
        assert self.fn([]) is False

    def test_rejects_non_list(self):
        assert self.fn({"haru": "Hello!"}) is False

    def test_rejects_list_of_non_dicts(self):
        assert self.fn(["hello", "world"]) is False


# ---------------------------------------------------------------------------
# _extract_metadata_from_filename (via process_uploaded_transcript module)
# ---------------------------------------------------------------------------

class TestExtractMetadataFromFilename:

    @pytest.fixture(autouse=True)
    def import_helper(self):
        # module already imported at top of file — no local import needed
        self.fn = process_uploaded_transcript._extract_metadata_from_filename

    def test_standard_format(self):
        pid, name = self.fn("P1_Bob.json")
        assert pid == "P1"
        assert name == "Bob"

    def test_larger_participant_number(self):
        pid, name = self.fn("P12_Alice.json")
        assert pid == "P12"
        assert name == "Alice"

    def test_name_with_underscore(self):
        pid, name = self.fn("P3_Mary_Jane.json")
        assert pid == "P3"
        assert name == "Mary_Jane"

    def test_no_match_returns_none_pair(self):
        pid, name = self.fn("conversation.json")
        assert pid is None
        assert name is None

    def test_lowercase_p_is_uppercased(self):
        pid, name = self.fn("p2_dave.json")
        assert pid == "P2"
        assert name == "dave"

    def test_txt_extension_also_works(self):
        pid, name = self.fn("P4_Emma.txt")
        assert pid == "P4"
        assert name == "Emma"

    def test_missing_name_part_no_match(self):
        pid, name = self.fn("P1_.json")
        assert pid is None
        assert name is None


# ---------------------------------------------------------------------------
# _assign_topics — trigger-phrase based topic detection
# ---------------------------------------------------------------------------

@pytest.fixture
def topic_converter():
    """Converter in script mode — only trigger-phrase _assign_topics logic is exercised."""
    return RawTranscriptConverter(topic_method="script")


class TestAssignTopics:

    def test_intro_trigger_on_haru_greeting(self, topic_converter):
        topics = topic_converter._assign_topics(["Hi, my name is Haru!"])
        assert topics[0] == "Intro"

    def test_food_trigger_advances_to_favourites(self, topic_converter):
        sentences = [
            "Hi, my name is Haru!",
            "My name is Alice.",
            "What is your favourite food?",
            "I like pizza.",
        ]
        topics = topic_converter._assign_topics(sentences)
        assert topics[2] == "Favourites"
        assert topics[3] == "Favourites"

    def test_books_trigger_also_maps_to_favourites(self, topic_converter):
        topics = topic_converter._assign_topics(
            ["Do you have a favourite book?", "I love Harry Potter."]
        )
        assert topics[0] == "Favourites"

    def test_holidays_trigger_maps_to_holidays(self, topic_converter):
        topics = topic_converter._assign_topics(
            ["What is your favourite holiday?", "Christmas!"]
        )
        assert topics[0] == "Holidays"

    def test_movies_trigger_also_maps_to_favourites(self, topic_converter):
        topics = topic_converter._assign_topics(
            ["How about your favourite movie?", "I love Moana."]
        )
        assert topics[0] == "Favourites"

    def test_favourites_advances_to_holidays_then_daily_life(self, topic_converter):
        sentences = [
            "What is your favourite food?",
            "I like pizza.",
            "What is your favourite holiday?",
            "Christmas!",
            "Let's talk about your everyday life.",
            "I wake up early.",
        ]
        topics = topic_converter._assign_topics(sentences)
        assert topics[0] == "Favourites"
        assert topics[1] == "Favourites"
        assert topics[2] == "Holidays"
        assert topics[3] == "Holidays"
        assert topics[4] == "Daily Life"
        assert topics[5] == "Daily Life"

    def test_topic_never_goes_back(self, topic_converter):
        sentences = [
            "Let's talk about your everyday life.",
            "I wake up early.",
            "What is your favourite food?",  # Favourites comes before Daily Life — must NOT revert
        ]
        topics = topic_converter._assign_topics(sentences)
        assert topics[2] == "Daily Life"

    def test_sentences_before_any_trigger_default_to_intro(self, topic_converter):
        topics = topic_converter._assign_topics(["Hello.", "How are you?"])
        assert all(t == "Intro" for t in topics)

    def test_all_returned_topics_are_valid_labels(self, topic_converter):
        sentences = [
            "What is your favourite food?",
            "I like pizza.",
            "Let's talk about your everyday life.",
            "I wake up at 7.",
        ]
        topics = topic_converter._assign_topics(sentences)
        for t in topics:
            assert t in RawTranscriptConverter.TOPIC_LABELS

    def test_memories_trigger_summer(self, topic_converter):
        topics = topic_converter._assign_topics(
            ["What did you do last summer?", "I went swimming."]
        )
        assert topics[0] == "Memories"

    def test_memories_trigger_long_walk(self, topic_converter):
        topics = topic_converter._assign_topics(
            ["Can you remember the last time you went for a long walk?", "Yes!"]
        )
        assert topics[0] == "Memories"

    def test_gifts_trigger(self, topic_converter):
        topics = topic_converter._assign_topics(
            ["I love a little gift! Have you ever received a thoughtful gift?", "Yes!"]
        )
        assert topics[0] == "Gifts"

    def test_imagination_trigger_famous(self, topic_converter):
        topics = topic_converter._assign_topics(
            ["Would you like to be famous someday?", "Maybe!"]
        )
        assert topics[0] == "Imagination"

    def test_imagination_trigger_superpower(self, topic_converter):
        topics = topic_converter._assign_topics(
            ["If you could wake up tomorrow with a superpower, what would it be?", "Flying!"]
        )
        assert topics[0] == "Imagination"

    def test_imagination_covers_wishes(self, topic_converter):
        topics = topic_converter._assign_topics(
            ["If you had one wish that could come true, what would you wish for?", "World peace."]
        )
        assert topics[0] == "Imagination"

    def test_imagination_covers_ice_cream(self, topic_converter):
        topics = topic_converter._assign_topics(
            ["If you could invent a brand new ice cream flavour, what would it be?", "Flower!"]
        )
        assert topics[0] == "Imagination"

    def test_empty_returns_empty(self, topic_converter):
        assert topic_converter._assign_topics([]) == []


# ---------------------------------------------------------------------------
# _assign_topics — local model clustering (embedding model and Ollama mocked)
# ---------------------------------------------------------------------------

@pytest.fixture
def model_converter():
    c = RawTranscriptConverter(topic_method="model", max_topics=9)
    # Pairs 0-1 point one way, pairs 2-3 point the other → one topic break.
    c._embedding_model = MagicMock()
    c._embedding_model.encode = MagicMock(
        return_value=[[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.1, 1.0]]
    )
    return c


def _ollama_response(text):
    response = MagicMock()
    response.json.return_value = {"response": text}
    response.raise_for_status.return_value = None
    return response


class TestAssignTopicsByModel:

    SENTENCES = [
        "What is your favourite food?", "Pizza.",
        "What is your favourite book?", "Harry Potter.",
        "What do you do at school?", "Maths.",
        "What is your favourite subject?", "Art.",
    ]

    def test_segments_get_one_label_each(self, model_converter, monkeypatch):
        labels = iter(["Favourites", "School"])
        monkeypatch.setattr(
            "raw_transcript_converter.requests.post",
            lambda *a, **k: _ollama_response(next(labels)),
        )
        topics = model_converter._assign_topics(self.SENTENCES)
        assert topics == ["Favourites"] * 4 + ["School"] * 4

    def test_uses_configured_model_and_host(self, monkeypatch):
        c = RawTranscriptConverter(topic_model="qwen2.5:7b", ollama_host="http://gpu-box:11434/")
        calls = []

        def fake_post(url, json, timeout):
            calls.append((url, json["model"]))
            return _ollama_response("Hobbies")

        monkeypatch.setattr("raw_transcript_converter.requests.post", fake_post)
        c._make_short_label("Do you like drawing? Yes.")
        assert calls == [("http://gpu-box:11434/api/generate", "qwen2.5:7b")]

    def test_env_vars_select_models(self, monkeypatch):
        monkeypatch.setenv("HARU_TOPIC_MODEL", "llama3.1:8b")
        monkeypatch.setenv("HARU_EMBEDDING_MODEL", "all-mpnet-base-v2")
        monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11434")
        c = RawTranscriptConverter()
        assert c.topic_model == "llama3.1:8b"
        assert c.embedding_model_name == "all-mpnet-base-v2"
        assert c.ollama_host == "http://127.0.0.1:11434"

    def test_falls_back_to_keywords_when_ollama_down(self, model_converter, monkeypatch):
        def refuse(*a, **k):
            raise ConnectionError("refused")

        monkeypatch.setattr("raw_transcript_converter.requests.post", refuse)
        topics = model_converter._assign_topics(self.SENTENCES)
        assert len(topics) == len(self.SENTENCES)
        assert all(t and t != "General" for t in topics)

    def test_long_label_is_trimmed(self, model_converter, monkeypatch):
        monkeypatch.setattr(
            "raw_transcript_converter.requests.post",
            lambda *a, **k: _ollama_response("Extracurricular Activities and more"),
        )
        assert model_converter._make_short_label("text") == "Extracurricular"

    def test_two_word_label_within_limit_is_kept(self, model_converter, monkeypatch):
        monkeypatch.setattr(
            "raw_transcript_converter.requests.post",
            lambda *a, **k: _ollama_response("Summer Holidays"),
        )
        assert model_converter._make_short_label("text") == "Summer Holidays"

    def test_max_label_chars_configurable(self, monkeypatch):
        monkeypatch.setenv("HARU_MAX_LABEL_CHARS", "12")
        c = RawTranscriptConverter()
        monkeypatch.setattr(
            "raw_transcript_converter.requests.post",
            lambda *a, **k: _ollama_response("Summer Holidays"),
        )
        assert c._make_short_label("text") == "Summer"

    def test_label_override_applied(self, model_converter, monkeypatch):
        monkeypatch.setattr(
            "raw_transcript_converter.requests.post",
            lambda *a, **k: _ollama_response("Name."),
        )
        assert model_converter._make_short_label("What is your name?") == "Intro"

    def test_empty_returns_empty(self, model_converter):
        assert model_converter._assign_topics([]) == []

    def test_invalid_topic_method_raises(self):
        with pytest.raises(ValueError):
            RawTranscriptConverter(topic_method="magic")


class TestTopicModelSpeed:

    def test_default_host_avoids_localhost(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        assert RawTranscriptConverter().ollama_host == "http://127.0.0.1:11434"

    def test_keep_alive_sent(self, model_converter, monkeypatch):
        sent = []
        monkeypatch.setattr(
            "raw_transcript_converter.requests.post",
            lambda url, json, timeout: sent.append(json) or _ollama_response("Pets"),
        )
        model_converter._make_short_label("Do you have a dog? Yes.")
        assert sent[0]["keep_alive"] == RawTranscriptConverter.OLLAMA_KEEP_ALIVE

    def test_stops_calling_ollama_after_first_failure(self, model_converter, monkeypatch):
        calls = []

        def refuse(*a, **k):
            calls.append(1)
            raise ConnectionError("refused")

        monkeypatch.setattr("raw_transcript_converter.requests.post", refuse)
        model_converter._assign_topics(TestAssignTopicsByModel.SENTENCES)  # 2 segments
        assert len(calls) == 1

    def test_retries_ollama_on_next_transcript(self, model_converter, monkeypatch):
        def refuse(*a, **k):
            raise ConnectionError("refused")

        monkeypatch.setattr("raw_transcript_converter.requests.post", refuse)
        model_converter._assign_topics(TestAssignTopicsByModel.SENTENCES)

        monkeypatch.setattr(
            "raw_transcript_converter.requests.post",
            lambda *a, **k: _ollama_response("Favourites"),
        )
        topics = model_converter._assign_topics(TestAssignTopicsByModel.SENTENCES)
        assert topics[0] == "Favourites"


def _fake_encoder(exchange_vectors, line_vectors=None):
    """
    First call (exchanges) returns exchange_vectors. Later calls (single lines, used to decide
    who changed the topic) look each line up in line_vectors, defaulting to one shared vector,
    so equal similarity keeps the topic change on the robot.
    """
    calls = []

    def encode(texts):
        calls.append(texts)
        if len(calls) == 1:
            return exchange_vectors
        return [(line_vectors or {}).get(t, [1.0, 1.0, 1.0]) for t in texts]

    encoder = MagicMock()
    encoder.encode = MagicMock(side_effect=encode)
    return encoder


class TestLinearTopics:
    """Each topic must appear once, in order, as one contiguous block."""

    TURNS = ["robot", "user"] * 4

    @staticmethod
    def _converter_with_segments(boundaries, n_pairs=4):
        # Orthogonal embeddings at each boundary force a new segment there.
        c = RawTranscriptConverter(topic_method="model")
        vecs, axis = [], 0
        for i in range(n_pairs):
            if i in boundaries:
                axis += 1
            vecs.append([1.0 if j == axis else 0.0 for j in range(n_pairs + 1)])
        c._embedding_model = _fake_encoder(vecs)
        return c

    # 8 exchanges (16 lines); orthogonal embeddings force splits before exchanges 2, 4 and 6.
    FOUR_TOPIC_SENTENCES = [f"line {i}" for i in range(16)]
    FOUR_TOPIC_TURNS = ["robot", "user"] * 8

    def _four_topics(self, monkeypatch, labels):
        c = self._converter_with_segments({2, 4, 6}, n_pairs=8)
        c.max_topics = 4
        labels = iter(labels)
        monkeypatch.setattr(
            "raw_transcript_converter.requests.post",
            lambda *a, **k: _ollama_response(next(labels)),
        )
        return c

    def test_repeated_label_continues_previous_topic(self, monkeypatch):
        c = self._four_topics(monkeypatch, ["Intro", "Pets", "Intro", "School"])
        topics = c._assign_topics(self.FOUR_TOPIC_SENTENCES, self.FOUR_TOPIC_TURNS)

        # "Intro" came back for segment 3, so it stays on "Pets" instead of looping back.
        assert topics == ["Intro"] * 4 + ["Pets"] * 8 + ["School"] * 4

    def test_each_label_forms_one_contiguous_block(self, monkeypatch):
        c = self._four_topics(monkeypatch, ["Fun", "Fun", "Games", "Fun"])
        topics = c._assign_topics(self.FOUR_TOPIC_SENTENCES, self.FOUR_TOPIC_TURNS)

        blocks = [t for i, t in enumerate(topics) if i == 0 or topics[i - 1] != t]
        assert len(blocks) == len(set(blocks))

    def test_used_labels_sent_to_model(self, monkeypatch):
        c = self._converter_with_segments({2, 4, 6}, n_pairs=8)
        c.max_topics = 4
        prompts = []

        def fake_post(url, json, timeout):
            prompts.append(json["prompt"])
            return _ollama_response(["A", "B", "C", "D"][len(prompts) - 1])

        monkeypatch.setattr("raw_transcript_converter.requests.post", fake_post)
        c._assign_topics(self.FOUR_TOPIC_SENTENCES, self.FOUR_TOPIC_TURNS)

        assert "Do NOT use" not in prompts[0]
        assert "Do NOT use any of them: A, B, C" in prompts[3]

    # Four exchanges; the clustering puts a topic break before "Do you have pets?".
    WHO_SENTENCES = ["Hi, I'm Haru! What food do you like?", "Pasta.",
                     "What is your favourite food?", "Pizza! Also my dog is called Max.",
                     "Do you have pets?", "Yes, a dog.",
                     "What do you do at school?", "Art."]
    WHO_TURNS = ["robot", "user"] * 4
    WHO_EXCHANGES = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]

    def _who_converter(self, monkeypatch, line_vectors):
        c = RawTranscriptConverter(topic_method="model")
        c._embedding_model = _fake_encoder(self.WHO_EXCHANGES, line_vectors)
        labels = iter(["Food", "Pets"])
        monkeypatch.setattr(
            "raw_transcript_converter.requests.post",
            lambda *a, **k: _ollama_response(next(labels)),
        )
        return c

    def test_robot_changes_topic_when_reply_answers_question(self, monkeypatch):
        c = self._who_converter(monkeypatch, {
            "What is your favourite food?": [1.0, 0.0, 0.0],
            "Pizza! Also my dog is called Max.": [1.0, 0.1, 0.0],
            "Do you have pets?": [0.0, 1.0, 0.0],
        })
        topics = c._assign_topics(self.WHO_SENTENCES, self.WHO_TURNS)

        # Topic changes on Haru's line, so the diagram draws the change from the robot.
        assert topics == ["Food"] * 4 + ["Pets"] * 4

    def test_user_changes_topic_when_reply_leads_into_next_topic(self, monkeypatch):
        c = self._who_converter(monkeypatch, {
            "What is your favourite food?": [1.0, 0.0, 0.0],
            "Pizza! Also my dog is called Max.": [0.1, 1.0, 0.0],
            "Do you have pets?": [0.0, 1.0, 0.0],
        })
        topics = c._assign_topics(self.WHO_SENTENCES, self.WHO_TURNS)

        # The child brought up the dog and Haru followed: the change starts on the user's line.
        assert topics == ["Food"] * 3 + ["Pets"] * 5

    def test_previous_topic_keeps_at_least_one_line(self, monkeypatch):
        c = RawTranscriptConverter(topic_method="model")
        segments = [[0, 1], [2, 3]]
        c._embedding_model = _fake_encoder(None, {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.0, 1.0]})
        c._embedding_model.encode([])  # consume the exchange call
        c._assign_topic_changes_to_speaker(["a", "b", "c", "d"], ["user", "user", "robot", "user"], segments)
        assert segments == [[0, 1], [2, 3]]

    def test_exchanges_group_robot_with_following_user_lines(self):
        c = RawTranscriptConverter(topic_method="model")
        turns = ["robot", "user", "user", "robot", "robot", "user"]
        assert c._group_exchanges(["x"] * 6, turns) == [[0, 1, 2], [3], [4, 5]]

    def test_leading_user_line_joins_first_exchange(self):
        c = RawTranscriptConverter(topic_method="model")
        assert c._group_exchanges(["x"] * 3, ["user", "robot", "user"]) == [[0, 1, 2]]

    def test_short_reply_never_changes_topic(self, monkeypatch):
        sentences = list(self.WHO_SENTENCES)
        sentences[3] = "Yes."
        c = self._who_converter(monkeypatch, {
            "What is your favourite food?": [1.0, 0.0, 0.0],
            "Yes.": [0.0, 1.0, 0.0],
            "Do you have pets?": [0.0, 1.0, 0.0],
        })
        topics = c._assign_topics(sentences, self.WHO_TURNS)
        assert topics[3] == "Food"

    def test_question_part_skips_echo(self):
        q = RawTranscriptConverter._question_part
        assert q("That's cool! Pierogies are yummy.\n\nDo you have a favourite book?") == \
            "Do you have a favourite book?"
        assert q("That's so sweet! What kind of dog is he?") == "What kind of dog is he?"
        assert q("Hello") == "Hello"

    def test_single_line_topic_joins_next_topic(self):
        segments = [[0, 1], [2], [3, 4]]
        RawTranscriptConverter._merge_single_line_segments(segments)
        assert segments == [[0, 1], [2, 3, 4]]

    def test_single_line_at_end_joins_previous_topic(self):
        segments = [[0, 1], [2]]
        RawTranscriptConverter._merge_single_line_segments(segments)
        assert segments == [[0, 1, 2]]


class TestSplitAtBiggestDrops:

    def _split(self, similarities, max_topics=9):
        c = RawTranscriptConverter(topic_method="model", max_topics=max_topics)
        return c._split_at_biggest_drops(similarities, len(similarities) + 1)

    def test_full_session_gets_max_topics_even_when_all_similar(self):
        # A smooth conversation: every neighbour is fairly similar (no fixed cutoff would split).
        sims = [0.6, 0.55, 0.5, 0.62, 0.45, 0.58, 0.52, 0.61, 0.44, 0.57,
                0.6, 0.47, 0.63, 0.5, 0.59, 0.46, 0.6, 0.62, 0.43, 0.6]
        segments = self._split(sims)
        assert len(segments) == 9

    def test_splits_at_lowest_similarities(self):
        sims = [0.9, 0.1, 0.9, 0.9, 0.2, 0.9, 0.9]  # drops before exchanges 2 and 5
        segments = self._split(sims, max_topics=3)
        assert segments == [[0, 1], [2, 3, 4], [5, 6, 7]]

    def test_splits_never_adjacent(self):
        sims = [0.9, 0.1, 0.15, 0.9, 0.9, 0.9, 0.3, 0.9]  # 2nd-lowest is next to the lowest
        segments = self._split(sims, max_topics=3)
        assert all(len(seg) >= 2 for seg in segments)
        # Lowest is before 2; before 3 is adjacent so it's skipped; next lowest is before 7.
        assert segments == [[0, 1], [2, 3, 4, 5, 6], [7, 8]]

    def test_short_transcript_gets_fewer_topics(self):
        assert len(self._split([0.1, 0.9, 0.1])) == 2  # 4 exchanges -> 2 topics
        assert self._split([]) == [[0]]

    def test_covers_every_exchange_once(self):
        sims = [0.3, 0.8, 0.2, 0.7, 0.9, 0.1, 0.6, 0.4, 0.5, 0.35, 0.75]
        segments = self._split(sims, max_topics=5)
        assert [i for seg in segments for i in seg] == list(range(len(sims) + 1))


class TestSystemMarkers:

    def test_marker_detection(self):
        from raw_transcript_converter import is_system_marker
        assert is_system_marker("[START]")
        assert is_system_marker("  [END] ")
        assert not is_system_marker("Hi [child's name]!")
        assert not is_system_marker("[yes]")
        assert not is_system_marker("[laughs]")
        assert not is_system_marker("START")
        assert is_system_marker("[start]")
        assert is_system_marker("[ Start ]")
        assert is_system_marker("[STOP]")

    def test_start_line_dropped_from_json_transcript(self, converter):
        data = [
            {"time": "2026-05-06T18:00:00+00:00", "haru": "Hi, my name is Haru!", "user": "[START]"},
            {"time": "2026-05-06T18:00:05+00:00", "haru": "Nice to meet you!", "user": "I'm Bob."},
        ]
        result = converter.convert_json_transcript(data)
        assert [u["sentence"] for u in result] == ["Hi, my name is Haru!", "I'm Bob.", "Nice to meet you!"]
        assert [u["idx"] for u in result] == [0, 1, 2]

    def test_start_line_dropped_from_text_transcript(self, converter):
        text = "{'user': '[START]'}\n{'robot': 'Hello!'}\n{'user': 'Hi.'}"
        result = converter.convert_text(text)
        assert [u["sentence"] for u in result] == ["Hello!", "Hi."]


class TestFilenameWithoutP:

    def test_number_without_p(self):
        assert process_uploaded_transcript._extract_metadata_from_filename("16_Raiken.json") == ("P16", "Raiken")

    def test_number_without_p_dash(self):
        assert process_uploaded_transcript._extract_metadata_from_filename("14-Wesley.json") == ("P14", "Wesley")

    def test_split_session_without_extension(self):
        f = process_uploaded_transcript._extract_metadata_from_filename
        assert f("P19.1-Alex") == ("P19.1", "Alex")
        assert f("P16-Ryken") == ("P16", "Ryken")
        assert f("P19.2-Alex.json") == ("P19.2", "Alex")

    def test_name_then_number(self):
        f = process_uploaded_transcript._extract_metadata_from_filename
        assert f("Lucy5.1") == ("P5.1", "Lucy")
        assert f("Lucy5.2") == ("P5.2", "Lucy")
