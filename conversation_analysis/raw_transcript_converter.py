import re
import requests
from datetime import datetime, timedelta

from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline


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

    LABEL_OVERRIDES = {
        "control": "AI Future",
        "name": "Intro",
        "talk": "Intro",
    }

    def __init__(self, transcript_date=None, max_topics=9, ollama_model="llama3.2:1b"):
        self.transcript_date = transcript_date or datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        self.max_topics = max_topics
        self.ollama_model = ollama_model
        self.vader = SentimentIntensityAnalyzer()

        self.emotion_model = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=1,
        )

        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

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
                # Stop after we have both or hit the transcript content
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

        sentences = [entry["sentence"] for entry in entries]
        topics = self._assign_topics(sentences)

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

        return conversation

    def convert_json_transcript(self, json_data):
        """
        Parses the Haru JSON format:
        [{"time": "2026-05-06T18:00:00+00:00", "haru": "...", "user": "..."}, ...]

        Each entry always has a haru utterance; user is optional. When both are
        present, haru is listed first (as it appears in the conversation), followed
        by the user response one second later.
        """
        entries = []
        timestamps = []

        for item in json_data:
            haru_text = item.get("haru", "").strip()
            user_text = item.get("user", "").strip()
            time_str = item.get("time", "")

            try:
                ts = datetime.fromisoformat(time_str)
                ts_naive = ts.replace(tzinfo=None)
            except (ValueError, TypeError):
                ts_naive = datetime.strptime(self.transcript_date, "%Y-%m-%d %H:%M:%S")

            if haru_text:
                entries.append({"speaker": "haru", "sentence": haru_text})
                timestamps.append(ts_naive.strftime("%Y-%m-%d %H:%M:%S"))

            if user_text:
                entries.append({"speaker": "user", "sentence": user_text})
                timestamps.append((ts_naive + timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S"))

        if not entries:
            raise ValueError("No transcript entries found in JSON data.")

        sentences = [e["sentence"] for e in entries]
        topics = self._assign_topics(sentences)

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

        return conversation

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

            if sentence:
                entries.append({"speaker": speaker, "sentence": sentence})

        return entries

    def _normalize_turn(self, speaker):
        speaker = speaker.lower()

        if speaker in ["robot", "haru", "assistant", "bot", "interviewer"]:
            return "robot"

        return "user"

    def _get_emotion(self, sentence):
        result = self.emotion_model(sentence)[0][0]
        return result["label"].lower(), float(result["score"])

    def _get_sentiment(self, sentence):
        result = self.emotion_model(sentence)[0][0]
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

    def _assign_topics(self, sentences):
        """
        Sequential topic assignment using Q/A pairs.
        Reads transcript in order, groups nearby Q/A pairs into topic segments,
        and assigns one short topic label to all utterances in that segment.
        """

        if not sentences:
            return []

        # 1. Group transcript into Q/A pairs: robot + user
        pairs = []
        pair_indices = []

        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences):
                pair_text = sentences[i] + " " + sentences[i + 1]
                pairs.append(pair_text)
                pair_indices.append([i, i + 1])
                i += 2
            else:
                pairs.append(sentences[i])
                pair_indices.append([i])
                i += 1

        if len(pairs) == 1:
            return [self._make_short_label(pairs[0]) for _ in sentences]

        # 2. Embed Q/A pairs instead of individual utterances
        embeddings = self.embedding_model.encode(pairs)

        segments = []
        current_segment = [0]

        # Adjusting similarity_threshold makes the topic classification more or less
        # sensitive to topic shifts. 0.38 is a good starting point for short Q/A pairs,
        # but you may want to adjust it based on your specific transcripts.
        similarity_threshold = 0.38

        # 3. Detect topic changes between neighboring Q/A pairs
        for pair_idx in range(1, len(pairs)):
            similarity = self._cosine_similarity(
                embeddings[pair_idx - 1],
                embeddings[pair_idx]
            )

            if similarity < similarity_threshold:
                segments.append(current_segment)
                current_segment = [pair_idx]
            else:
                current_segment.append(pair_idx)

        segments.append(current_segment)

        # 4. Merge neighboring segments until there are max_topics or fewer
        while len(segments) > self.max_topics:
            smallest_index = min(range(len(segments)), key=lambda idx: len(segments[idx]))

            if smallest_index == 0:
                segments[1] = segments[0] + segments[1]
                segments.pop(0)
            else:
                segments[smallest_index - 1] = (
                    segments[smallest_index - 1] + segments[smallest_index]
                )
                segments.pop(smallest_index)

        # 5. Assign one label to all utterances in each segment
        topics = ["General"] * len(sentences)

        for segment in segments:
            segment_pair_texts = [pairs[pair_idx] for pair_idx in segment]
            segment_text = " ".join(segment_pair_texts)

            label = self._make_short_label(segment_text)

            for pair_idx in segment:
                for sentence_idx in pair_indices[pair_idx]:
                    topics[sentence_idx] = label

        return topics

    def _cosine_similarity(self, vector_a, vector_b):
        dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
        norm_a = sum(a * a for a in vector_a) ** 0.5
        norm_b = sum(b * b for b in vector_b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0

        return dot_product / (norm_a * norm_b)

    def _make_short_label(self, text):
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) > 2:
            excerpt = sentences[0] + '. ' + sentences[-1]
        else:
            excerpt = text

        prompt = (
            "Read the following conversation excerpt and respond with ONLY a 1 to 2 word "
            "high-level theme label. Think about the broad category this conversation belongs to, "
            "not the specific words used. For example, talking about food, books, or movies "
            "would all be 'Favourites'. Talking about school or subjects would be 'School'. "
            "No punctuation, no explanation, just 1 or 2 words maximum.\n\n"
            f"Excerpt: {excerpt}"
        )

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 20,
                    },
                },
                timeout=15,
            )
            response.raise_for_status()
            label = response.json().get("response", "").strip()

            label = label.splitlines()[0].strip(" .,\"'")
            label = " ".join(label.split()[:2])

            # Hard cap — if label is still too long, fall back to first word only
            if len(label) > 12:
                label = label.split()[0]

            # Apply label overrides for known bad labels
            if label.lower() in self.LABEL_OVERRIDES:
                return self.LABEL_OVERRIDES[label.lower()]

            return label if label else self._keyword_fallback(text)

        except Exception as e:
            print(f"[Ollama fallback] {e}")
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
        if len(result) > 12:
            result = meaningful_words[0]

        return result

