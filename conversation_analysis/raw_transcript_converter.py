import os
import re
import threading
import requests
from datetime import datetime, timedelta

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

from llm_review import LLMReviewer

# Whole-line system commands from the robot software, e.g. "[START]". These aren't speech,
# so they're dropped. Matches any all-caps bracketed word ("[START]", "[END]") and the known
# commands in any case or spacing ("[start]", "[ Start ]"). Bracketed text inside a real
# sentence ("Hi [child's name]!") and lower-case notes like "[laughs]" are kept.
SYSTEM_MARKER = re.compile(
    r"^\[\s*(?:[A-Z][A-Z _-]*|(?i:start|end|stop|begin|restart))\s*\]$"
)


def is_system_marker(text):
    return bool(SYSTEM_MARKER.match(str(text).strip()))


class RawTranscriptConverter:
    """
    Converts a labeled raw transcript text file into the JSON structure
    expected by the Haru Chat visualization pipeline.

    Expected transcript format:

    {P1}
    {John}
    {'robot': 'Tell me about something you enjoyed in AI Club.'}
    {'user': 'I liked learning about AI ethics.'}
    """

    # Ordered topic labels that follow Haru's fixed conversation script.
    TOPIC_LABELS = [
        "Intro", "Favourites", "Holidays", "Daily Life", "Fun",
        "School", "Memories", "Gifts", "Imagination",
    ]

    # Trigger phrases keyed by topic. When any phrase appears (case-insensitive)
    # in a sentence, the topic advances to that label and stays there until the
    # next trigger. Phrases are matched as substrings so they must be specific
    # enough not to appear in a child's response.
    TOPIC_TRIGGERS = {
        "Intro": [
            "my name is haru",
            "what is your name",
        ],
        "Favourites": [
            "favourite things",
            "favourite food",
            "favorite food",
            "favourite book",
            "favorite book",
            "favourite movie",
            "favorite movie",
        ],
        "Holidays": [
            "favourite holiday",
            "favorite holiday",
        ],
        "Daily Life": [
            "everyday life",
            "typical day",
            "go to bed early or stay up late",
        ],
        "Fun": [
            "what do you like to do for fun",
            "do for fun",
        ],
        "School": [
            "favourite subject",
            "favorite subject",
            "let's talk about school",
            "talk about school",
            "what did your school do",
        ],
        "Memories": [
            "past experiences",
            "what did you do last summer",
            "last time you went for a long walk",
        ],
        "Gifts": [
            "thoughtful gift",
            "love a little gift",
        ],
        "Imagination": [
            "use your imagination",
            "famous someday",
            "would you like to be famous",
            "plan your perfect day",
            "wake up tomorrow with a superpower",
            "one wish that could come true",
            "invent a brand new ice cream",
        ],
    }

    # SMALL-MODEL WORKAROUND: known bad labels from llama3.2:1b, remapped to something useful.
    # Only fires when the whole label is exactly one of these words (case-insensitive),
    # so "Name Origins" or "Self Control" pass through untouched. "control" -> "AI Future"
    # is the riskiest: a segment genuinely about self-control would be mislabelled.
    # To remove: set this to {} (keeps the lookup in _make_short_label harmless), and
    # delete test_label_override_applied in tests/test_json_transcript.py.
    LABEL_OVERRIDES = {
        "control": "AI Future",
        "name": "Intro",
        "talk": "Intro",
    }

    # Local model settings. Each can be overridden by a constructor argument or
    # an environment variable, so swapping to a better model needs no code change:
    #   HARU_TOPIC_MODEL      Ollama model used to name each topic segment
    #   HARU_EMBEDDING_MODEL  sentence-transformers model used to find topic shifts
    #   OLLAMA_HOST           Ollama server URL
    #   HARU_MAX_LABEL_CHARS  longest topic label kept before trimming to one word
    DEFAULT_TOPIC_MODEL = "gemma4:e4b"
    DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    # 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 first, which Ollama doesn't
    # listen on, and waits ~2s for that to fail on every request.
    DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
    # How long Ollama keeps the topic model in memory after a request (its default is 5m).
    # Reloading a multi-GB model costs 10s+, so keep it warm between uploads.
    OLLAMA_KEEP_ALIVE = "30m"
    DEFAULT_MAX_LABEL_CHARS = 20
    # A child's reply is only credited with changing the topic if it has at least this many
    # words and is this much more similar to the next topic's question than to the one it
    # answered. Raise the margin if children are credited too often, lower it if too rarely.
    MIN_TOPIC_CHANGE_WORDS = 3
    # Shortest topic allowed, in exchanges (robot line + reply). Short transcripts get fewer
    # topics; a full session (18+ exchanges) gets max_topics.
    MIN_EXCHANGES_PER_TOPIC = 2
    TOPIC_CHANGE_MARGIN = 0.05

    def __init__(
        self,
        transcript_date=None,
        topic_method="model",
        max_topics=9,
        topic_model=None,
        embedding_model=None,
        ollama_host=None,
        max_label_chars=None,
        correct_transcription=None,
        review_labels=None,
    ):
        """
        topic_method: "model"  — cluster with local embeddings, name topics with a local LLM
                      "script" — match Haru's fixed script trigger phrases (TOPIC_TRIGGERS)
        correct_transcription: let the topic model lightly fix obvious speech-to-text
                      mistakes in the child's lines (env HARU_CORRECT_TRANSCRIPTS=0 turns off)
        review_labels: let the topic model double-check emotion/sentiment labels
                      (env HARU_REVIEW_LABELS=0 turns off)
        """
        if topic_method not in ("model", "script"):
            raise ValueError(f"topic_method must be 'model' or 'script', got {topic_method!r}")

        self.transcript_date = transcript_date or datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        self.topic_method = topic_method
        self.max_topics = max_topics
        self.topic_model = topic_model or os.environ.get("HARU_TOPIC_MODEL", self.DEFAULT_TOPIC_MODEL)
        self.embedding_model_name = embedding_model or os.environ.get(
            "HARU_EMBEDDING_MODEL", self.DEFAULT_EMBEDDING_MODEL
        )
        host = ollama_host or os.environ.get("OLLAMA_HOST", self.DEFAULT_OLLAMA_HOST)
        if not host.startswith("http"):
            host = "http://" + host
        self.ollama_host = host.rstrip("/")
        self.max_label_chars = max_label_chars or int(
            os.environ.get("HARU_MAX_LABEL_CHARS", self.DEFAULT_MAX_LABEL_CHARS)
        )
        self._embedding_model = None
        self._embedding_lock = threading.Lock()
        # Set False after a failed Ollama request so the rest of the transcript skips
        # straight to keyword labels instead of waiting on each segment. Reset per transcript.
        self._topic_model_available = True
        self._emotion_cache = {}

        self.correct_transcription = _setting(correct_transcription, "HARU_CORRECT_TRANSCRIPTS")
        self.review_labels = _setting(review_labels, "HARU_REVIEW_LABELS")
        self.reviewer = LLMReviewer(self.ollama_host, self.topic_model, self.OLLAMA_KEEP_ALIVE)

        self.vader = SentimentIntensityAnalyzer()

        self.emotion_model = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=1,
        )

    @property
    def embedding_model(self):
        # Loaded on first use so script mode never downloads the embedding model.
        with self._embedding_lock:
            if self._embedding_model is None:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def extract_metadata(self, file_path):
        """
        Reads the top of the transcript file and extracts participant ID and
        child name from lines like {P1} and {John}.
        Returns (participant_id, child_name) or (None, None) if not found.
        """
        participant_id = None
        child_name = None

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                match = re.match(r'^\{(\w+)\}$', line)
                if match:
                    value = match.group(1)
                    if value.upper().startswith('P') and value[1:].isdigit():
                        participant_id = value.upper()
                    else:
                        child_name = value
                if participant_id and child_name:
                    break
                if line.startswith("{'") or line.startswith('{"'):
                    break

        return participant_id, child_name

    def convert_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            raw_text = file.read()

        return self.convert_text(raw_text)

    def convert_text(self, raw_text):
        entries = self._parse_labeled_transcript(raw_text)

        if not entries:
            raise ValueError(
                "No transcript entries found. Use labeled lines like 'robot: ...' and 'user: ...'."
            )

        self._correct_entries(entries)
        sentences = [entry["sentence"] for entry in entries]
        topics = self._assign_topics(sentences, [self._normalize_turn(e["speaker"]) for e in entries])

        conversation = []
        start_time = datetime.strptime(self.transcript_date, "%Y-%m-%d %H:%M:%S")

        for idx, entry in enumerate(entries):
            sentence = entry["sentence"]
            turn = self._normalize_turn(entry["speaker"])

            emotion_label, emotion_score = self._get_emotion(sentence)
            sentiment_label, sentiment_score = self._get_sentiment(sentence)

            conversation.append(
                {
                    "idx": idx,
                    "sentence": sentence,
                    "emotion_label": emotion_label,
                    "emotion_score": emotion_score,
                    "sentiment_label": sentiment_label,
                    "sentiment_score": sentiment_score,
                    "turn": turn,
                    "last_interaction": idx == len(entries) - 1,
                    "highlighted": False,
                    "intent": "default-intent",
                    "intent_category": "default",
                    "timestamp": (start_time + timedelta(seconds=idx * 2)).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "topic": topics[idx],
                    "type": "text",
                    "slots": [],
                }
            )
            if "original_sentence" in entry:
                conversation[-1]["original_sentence"] = entry["original_sentence"]

        self._review_conversation_labels(conversation)
        return conversation

    def convert_json_transcript(self, json_data):
        """
        Parses the Haru JSON format:
        [{"time": "2026-05-06T18:00:00+00:00", "haru": "...", "user": "..."}, ...]

        Each entry always has a haru utterance; user is optional. The first entry
        is Haru's opening greeting (no user input). For all subsequent entries,
        user speaks first and haru responds — so user is listed before haru.
        """
        entries = []
        timestamps = []

        for item in json_data:
            haru_text = item.get("haru", "").strip()
            user_text = item.get("user", "").strip()
            if is_system_marker(haru_text):
                haru_text = ""
            if is_system_marker(user_text):
                user_text = ""
            time_str = item.get("time", "")

            try:
                ts = datetime.fromisoformat(time_str)
                ts_naive = ts.replace(tzinfo=None)
            except (ValueError, TypeError):
                ts_naive = datetime.strptime(self.transcript_date, "%Y-%m-%d %H:%M:%S")

            if user_text:
                entries.append({"speaker": "user", "sentence": user_text})
                timestamps.append(ts_naive.strftime("%Y-%m-%d %H:%M:%S"))

            if haru_text:
                # Offset only when responding to a user turn; opening greetings use the exact timestamp.
                haru_ts = ts_naive + timedelta(seconds=1) if user_text else ts_naive
                entries.append({"speaker": "haru", "sentence": haru_text})
                timestamps.append(haru_ts.strftime("%Y-%m-%d %H:%M:%S"))

        if not entries:
            raise ValueError("No transcript entries found in JSON data.")

        self._correct_entries(entries)
        sentences = [e["sentence"] for e in entries]
        topics = self._assign_topics(sentences, [self._normalize_turn(e["speaker"]) for e in entries])

        conversation = []
        for idx, (entry, timestamp) in enumerate(zip(entries, timestamps)):
            sentence = entry["sentence"]
            turn = self._normalize_turn(entry["speaker"])
            emotion_label, emotion_score = self._get_emotion(sentence)
            sentiment_label, sentiment_score = self._get_sentiment(sentence)

            conversation.append({
                "idx": idx,
                "sentence": sentence,
                "emotion_label": emotion_label,
                "emotion_score": emotion_score,
                "sentiment_label": sentiment_label,
                "sentiment_score": sentiment_score,
                "turn": turn,
                "last_interaction": idx == len(entries) - 1,
                "highlighted": False,
                "intent": "default-intent",
                "intent_category": "default",
                "timestamp": timestamp,
                "topic": topics[idx],
                "type": "text",
                "slots": [],
            })
            if "original_sentence" in entry:
                conversation[-1]["original_sentence"] = entry["original_sentence"]

        self._review_conversation_labels(conversation)
        return conversation

    def _correct_entries(self, entries):
        """
        Step 1 of the LLM passes: lightly fixes speech-to-text mistakes in the child's lines,
        before emotion, sentiment and topics run on them. Haru's lines are its own generated
        text, so they're never changed. The original is kept as "original_sentence".
        """
        self.reviewer.reset()
        if not self.correct_transcription or self.topic_method != "model":
            return

        replies, positions, question = [], [], ""
        for idx, entry in enumerate(entries):
            if self._normalize_turn(entry["speaker"]) == "robot":
                question = entry["sentence"]
            else:
                replies.append((question, entry["sentence"]))
                positions.append(idx)

        for idx, corrected in zip(positions, self.reviewer.correct_transcription(replies)):
            if corrected:
                entries[idx]["original_sentence"] = entries[idx]["sentence"]
                entries[idx]["sentence"] = corrected

    def _review_conversation_labels(self, conversation):
        """
        Step 3 (after the emotion model and VADER): the topic model double-checks each label.
        Changed labels keep the model's original as "<field>_before_review".

        The topic model is the third opinion, not the final word:
        - Where the label and VADER already agree on the tone (joy + clearly positive text),
          the label stands. Small models otherwise flattened "Reading is wonderful!" to neutral.
        - It may never give a negative label to clearly positive text or the reverse (small
          models turned "Always have fun!" into anger).
        So it mainly settles lines where the emotion model and VADER disagree, or the tone
        is mixed.
        """
        if not self.review_labels or self.topic_method != "model":
            return

        reviewed = self.reviewer.review_labels([
            {
                "speaker": "child" if line["turn"] == "user" else "robot",
                "text": line["sentence"],
                "emotion": line["emotion_label"],
                "sentiment": line["sentiment_label"],
            }
            for line in conversation
        ])
        for line, (emotion, sentiment) in zip(conversation, reviewed):
            # One- or two-word lines ("Aria", "Yes.") carry too little to judge; keep them.
            if len(line["sentence"].split()) < 3:
                continue
            tone = self._vader_tone(line["sentence"])
            if tone and _polarity(line["emotion_label"]) == tone or _polarity(emotion) * tone < 0:
                emotion = line["emotion_label"]
            if tone and _polarity(line["sentiment_label"]) == tone or _polarity(sentiment) * tone < 0:
                sentiment = line["sentiment_label"]

            if emotion != line["emotion_label"]:
                line["emotion_label_before_review"] = line["emotion_label"]
                line["emotion_label"] = emotion
            if sentiment != line["sentiment_label"]:
                line["sentiment_label_before_review"] = line["sentiment_label"]
                line["sentiment_label"] = sentiment

    # VADER compound score beyond which text counts as clearly positive or negative.
    CLEAR_TONE = 0.3

    def _vader_tone(self, sentence):
        compound = self.vader.polarity_scores(sentence)["compound"]
        if compound >= self.CLEAR_TONE:
            return 1
        if compound <= -self.CLEAR_TONE:
            return -1
        return 0

    def extract_metadata_from_json(self, json_data):
        """
        Tries to extract the child's name from Haru's closing statement,
        e.g. "Thank you, Bob! This has been so much fun..."
        Returns (None, child_name) — participant ID is not in this format.
        """
        child_name = None
        tail = json_data[-5:] if len(json_data) >= 5 else json_data
        for item in reversed(tail):
            haru_text = item.get("haru", "")
            match = re.search(r"[Tt]hank you,\s+([A-Z][a-z]+)!", haru_text)
            if match:
                child_name = match.group(1)
                break
        return None, child_name

    def _parse_labeled_transcript(self, raw_text):
        entries = []

        pattern = re.compile(
            r"\{\s*['\"](?P<speaker>haru|robot|user|child|participant|interviewer)['\"]\s*:\s*['\"](?P<sentence>.*?)['\"]\s*\}\s*,?",
            re.IGNORECASE | re.DOTALL,
        )

        for match in pattern.finditer(raw_text):
            speaker = match.group("speaker").strip()
            sentence = match.group("sentence").strip()

            if sentence and not is_system_marker(sentence):
                entries.append({"speaker": speaker, "sentence": sentence})

        return entries

    def _normalize_turn(self, speaker):
        speaker = speaker.lower()

        if speaker in ["robot", "haru", "assistant", "bot", "interviewer"]:
            return "robot"

        return "user"

    def _classify_emotion(self, sentence):
        # Emotion and sentiment both need this result; caching it halves the model calls.
        if sentence not in self._emotion_cache:
            self._emotion_cache[sentence] = self.emotion_model(sentence)[0][0]
        return self._emotion_cache[sentence]

    def _get_emotion(self, sentence):
        result = self._classify_emotion(sentence)
        return result["label"].lower(), float(result["score"])

    def _get_sentiment(self, sentence):
        result = self._classify_emotion(sentence)
        emotion = result["label"].lower()

        positive_emotions = {"joy", "surprise"}
        negative_emotions = {"anger", "disgust", "fear", "sadness"}

        if emotion in positive_emotions:
            label = "positive"
        elif emotion in negative_emotions:
            label = "negative"
        else:
            label = "neutral"

        compound = self.vader.polarity_scores(sentence)["compound"]
        score = round(abs(compound), 4)

        # VADER tiebreaker — override neutral if VADER detects strong sentiment
        if label == "neutral":
            if compound > 0.5:
                label = "positive"
            elif compound < -0.5:
                label = "negative"

        return label, score

    def _assign_topics(self, sentences, turns=None):
        """
        turns: "robot"/"user" per sentence. Used by model clustering to group exchanges and
        to work out whether the robot or the user changed each topic.
        """
        if self.topic_method == "script":
            return self._assign_topics_by_script(sentences)
        return self._assign_topics_by_model(sentences, turns)

    def _group_exchanges(self, sentences, turns):
        """
        Groups sentence indices into exchanges: a robot line plus the user replies after it.
        Clustering compares whole exchanges because a child's reply alone ("Pizza.") is too
        short to place. Without turns, falls back to fixed pairs of lines.
        """
        if turns is None:
            return [list(range(i, min(i + 2, len(sentences)))) for i in range(0, len(sentences), 2)]

        exchanges = []
        for idx, turn in enumerate(turns):
            # Lines before Haru's first line (e.g. a "[START]" marker) join the first exchange.
            if turn == "robot" and any(turns[i] == "robot" for i in range(idx)) or not exchanges:
                exchanges.append([idx])
            else:
                exchanges[-1].append(idx)
        return exchanges

    def _assign_topics_by_model(self, sentences, turns=None):
        """
        Sequential topic clustering using robot/user exchanges.
        Reads transcript in order, groups neighbouring exchanges into topic segments
        by embedding similarity, and names each segment with the local LLM.
        Topics are strictly linear: each label is used for one contiguous segment only.
        """
        if not sentences:
            return []

        self._topic_model_available = True

        # 1. Group transcript into exchanges: robot line + user reply
        pair_indices = self._group_exchanges(sentences, turns)
        pairs = [" ".join(sentences[i] for i in indices) for indices in pair_indices]

        if len(pairs) == 1:
            label = self._make_short_label(pairs[0])
            return [label] * len(sentences)

        # 2. Embed exchanges instead of individual utterances
        embeddings = self.embedding_model.encode(pairs)

        # 3. Split at the biggest drops in similarity between neighbouring exchanges.
        similarities = [
            self._cosine_similarity(embeddings[i - 1], embeddings[i]) for i in range(1, len(pairs))
        ]
        segments = self._split_at_biggest_drops(similarities, len(pairs))

        # 4. Turn exchange segments into sentence ranges, then decide who changed each topic.
        segment_sentences = [
            [sentence_idx for pair_idx in segment for sentence_idx in pair_indices[pair_idx]]
            for segment in segments
        ]
        if turns is not None:
            self._assign_topic_changes_to_speaker(sentences, turns, segment_sentences)

        # 5. A one-line topic would show as a bubble with only one speaker and an extra hop.
        self._merge_single_line_segments(segment_sentences)

        # 6. Label segments in order. A label may only be used once: the diagram draws one
        #    node per topic name, so a repeated name would loop the flow back to an earlier
        #    topic. If the model repeats a name anyway, that segment continues the previous topic.
        segment_labels = []
        used_labels = []

        for segment in segment_sentences:
            label = self._make_short_label(
                " ".join(sentences[i] for i in segment), avoid=used_labels
            )

            if used_labels and label.lower() in (u.lower() for u in used_labels):
                label = segment_labels[-1]
            else:
                used_labels.append(label)

            segment_labels.append(label)

        topics = ["General"] * len(sentences)

        for segment, label in zip(segment_sentences, segment_labels):
            for sentence_idx in segment:
                topics[sentence_idx] = label

        return topics

    def _split_at_biggest_drops(self, similarities, n_exchanges):
        """
        Returns segments (lists of exchange indices). Splits where neighbouring exchanges are
        least similar, as many times as the transcript's length allows, up to max_topics.

        This is relative to each transcript rather than a fixed similarity cutoff: Haru echoes
        the child, so smooth conversations can stay above any fixed cutoff for most of the
        session (one child's session got a single 46-line topic), while choppy ones fall below
        it too often. Two splits are never adjacent, so no topic is a single exchange.
        """
        target_topics = min(self.max_topics, max(1, n_exchanges // self.MIN_EXCHANGES_PER_TOPIC))

        breaks = []
        for pair_idx in sorted(range(1, n_exchanges), key=lambda i: similarities[i - 1]):
            if len(breaks) >= target_topics - 1:
                break
            if all(abs(pair_idx - b) >= 2 for b in breaks):
                breaks.append(pair_idx)

        bounds = [0] + sorted(breaks) + [n_exchanges]
        return [list(range(bounds[i], bounds[i + 1])) for i in range(len(bounds) - 1)]

    @staticmethod
    def _merge_single_line_segments(segment_sentences):
        """
        A one-line topic is almost always Haru's line introducing the next topic, so it joins
        the segment after it (the last segment joins the one before). Modifies in place.
        """
        k = 0
        while len(segment_sentences) > 1 and k < len(segment_sentences):
            if len(segment_sentences[k]) == 1:
                if k + 1 < len(segment_sentences):
                    segment_sentences[k + 1] = segment_sentences[k] + segment_sentences[k + 1]
                else:
                    segment_sentences[k - 1] = segment_sentences[k - 1] + segment_sentences[k]
                segment_sentences.pop(k)
            else:
                k += 1

    @staticmethod
    def _question_part(robot_line):
        """The part of a robot line that moves the conversation on, without the echo before it."""
        paragraphs = [p.strip() for p in robot_line.split("\n\n") if p.strip()]
        if len(paragraphs) > 1:
            return paragraphs[-1]
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", robot_line) if s.strip()]
        return sentences[-1] if sentences else robot_line

    def _assign_topic_changes_to_speaker(self, sentences, turns, segment_sentences):
        """
        Segments start on a robot line, which would always show Haru as the one who changed
        the topic. At each boundary, check the child's reply just before it: if it is clearly
        closer to the question that opens the next topic than to the question it answered,
        the child brought the new topic up, so it moves into the next segment. The diagram
        then draws the topic change from the user instead of the robot.
        Modifies segment_sentences in place.

        Haru usually echoes the child before asking something new ("That's cool! ... Do you
        have a favourite book?"), so only the question part of each robot line is compared;
        otherwise the echo makes every reply look like it led into the next topic.
        """
        candidates = []
        for k in range(1, len(segment_sentences)):
            previous = segment_sentences[k - 1]
            trailing_user = []
            for idx in reversed(previous):
                if turns[idx] != "user":
                    break
                trailing_user.insert(0, idx)

            # Keep at least two lines in the previous topic (a one-line topic gets merged away,
            # which would undo the change). Very short replies ("Yes.") can't introduce a
            # topic and give unreliable similarity scores.
            if (
                trailing_user
                and len(trailing_user) <= len(previous) - 2
                and len(sentences[trailing_user[-1]].split()) >= self.MIN_TOPIC_CHANGE_WORDS
            ):
                question = previous[len(previous) - len(trailing_user) - 1]
                candidates.append((k, trailing_user, question, segment_sentences[k][0]))

        if not candidates:
            return

        texts = {}
        for _, trailing_user, question, next_line in candidates:
            texts[trailing_user[-1]] = sentences[trailing_user[-1]]
            texts[question] = self._question_part(sentences[question])
            texts[next_line] = self._question_part(sentences[next_line])
        needed = sorted(texts)
        vectors = dict(zip(needed, self.embedding_model.encode([texts[i] for i in needed])))

        for k, trailing_user, question, next_line in candidates:
            reply = trailing_user[-1]
            similarity_to_question = self._cosine_similarity(vectors[reply], vectors[question])
            similarity_to_next = self._cosine_similarity(vectors[reply], vectors[next_line])

            if similarity_to_next > similarity_to_question + self.TOPIC_CHANGE_MARGIN:
                segment_sentences[k - 1] = segment_sentences[k - 1][: -len(trailing_user)]
                segment_sentences[k] = trailing_user + segment_sentences[k]

    def _cosine_similarity(self, vector_a, vector_b):
        dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
        norm_a = sum(a * a for a in vector_a) ** 0.5
        norm_b = sum(b * b for b in vector_b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0

        return dot_product / (norm_a * norm_b)

    def warm_up_topic_model(self):
        """Asks Ollama to load the topic model now, so the first label isn't slowed by loading."""
        requests.post(
            f"{self.ollama_host}/api/generate",
            json={"model": self.topic_model, "keep_alive": self.OLLAMA_KEEP_ALIVE},
            timeout=120,
        ).raise_for_status()

    def _make_short_label(self, text, avoid=()):
        if not self._topic_model_available:
            return self._keyword_fallback(text)

        # SMALL-MODEL WORKAROUND: only the first and last sentence of the segment are sent,
        # because a 1B model lost track of longer text. This hides most of the segment
        # from the model. A bigger model (e.g. gemma4:e4b) can read the whole segment:
        # to remove, replace these five lines with `excerpt = text`.
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) > 2:
            excerpt = sentences[0] + '. ' + sentences[-1]
        else:
            excerpt = text

        # The two examples steer labels toward broad categories ('Favourites', 'School').
        # Keep them if you like those names; edit them to change the naming style.
        prompt = (
            "Read the following conversation excerpt and respond with ONLY a 1 to 2 word "
            "high-level theme label. Think about the broad category this conversation belongs to, "
            "not the specific words used. For example, talking about food, books, or movies "
            "would all be 'Favourites'. Talking about school or subjects would be 'School'. "
            "No punctuation, no explanation, just 1 or 2 words maximum.\n\n"
        )
        if avoid:
            # Topics must not repeat (see _assign_topics_by_model), so steer away from used names.
            prompt += (
                "These labels are already used for earlier parts of the conversation. "
                f"Do NOT use any of them: {', '.join(avoid)}\n\n"
            )
        prompt += f"Excerpt: {excerpt}"

        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.topic_model,
                    "prompt": prompt,
                    "stream": False,
                    # Reasoning models would spend num_predict on thinking and return no label.
                    "think": False,
                    "keep_alive": self.OLLAMA_KEEP_ALIVE,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 20,
                    },
                },
                # Larger models can take a while to load on the first request.
                timeout=120,
            )
            response.raise_for_status()
            label = response.json().get("response", "").strip()

            label = label.splitlines()[0].strip(" .,\"'") if label else ""
            label = " ".join(label.split()[:2])

            # Hard cap — if label is still too long, fall back to first word only.
            # Keep this: it's cheap insurance even with good models (see max_label_chars).
            if len(label) > self.max_label_chars:
                label = label.split()[0]

            # SMALL-MODEL WORKAROUND lookup — harmless once LABEL_OVERRIDES is {}.
            if label.lower() in self.LABEL_OVERRIDES:
                return self.LABEL_OVERRIDES[label.lower()]

            return label if label else self._keyword_fallback(text)

        except Exception as e:
            print(f"[Topic model fallback] {self.topic_model} @ {self.ollama_host}: {e}")
            print("[Topic model fallback] Using keyword labels for the rest of this transcript.")
            self._topic_model_available = False
            return self._keyword_fallback(text)

    def _keyword_fallback(self, text):
        words = re.findall(r"[A-Za-z]+", text)

        stopwords = {
            "the", "and", "or", "but", "with", "about", "that", "this",
            "what", "when", "where", "why", "how", "you", "your", "they",
            "them", "are", "was", "were", "can", "could", "would", "should",
            "like", "really", "just", "something",
        }

        meaningful_words = [
            word.title()
            for word in words
            if word.lower() not in stopwords and len(word) > 2
        ]

        if not meaningful_words:
            return "General"

        result = " ".join(meaningful_words[:2])

        # Hard cap — if result is still too long, use first word only
        if len(result) > self.max_label_chars:
            result = meaningful_words[0]

        return result

    def _assign_topics_by_script(self, sentences):
        """
        Assigns topic labels by scanning each sentence for Haru's script trigger phrases.
        Topics only advance forward through TOPIC_LABELS — once a trigger is matched the
        topic cannot revert to an earlier one even if that trigger phrase appears again.
        """
        if not sentences:
            return []

        current_topic_idx = 0
        topics = []

        for sentence in sentences:
            sentence_lower = sentence.lower()

            # Scan forward from the current position for the next matching trigger.
            for idx in range(current_topic_idx, len(self.TOPIC_LABELS)):
                label = self.TOPIC_LABELS[idx]
                if any(phrase in sentence_lower for phrase in self.TOPIC_TRIGGERS.get(label, [])):
                    current_topic_idx = idx
                    break

            topics.append(self.TOPIC_LABELS[current_topic_idx])

        return topics


def _polarity(label):
    """+1 for positive labels, -1 for negative ones, 0 for neutral/surprise."""
    if label in ("joy", "positive"):
        return 1
    if label in ("anger", "disgust", "fear", "sadness", "negative"):
        return -1
    return 0


def _setting(value, env_name):
    """Constructor argument wins; otherwise the env var, where "0"/"false"/"off" disables."""
    if value is not None:
        return bool(value)
    return os.environ.get(env_name, "1").strip().lower() not in ("0", "false", "off", "no")
