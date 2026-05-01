import re
from datetime import datetime, timedelta

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from transformers import pipeline


class RawTranscriptConverter:
    """
    Converts a labeled raw transcript text file into the JSON structure
    expected by the Haru Chat visualization pipeline.

    Expected transcript format:

    robot: Tell me about something you enjoyed in AI Club.
    user: I liked learning about AI ethics.

    robot: What would you like to learn next?
    user: I want to learn how AI makes decisions.
    """

    def __init__(self, transcript_date=None, max_topics=10):
        self.transcript_date = transcript_date or datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        self.max_topics = max_topics

        self.emotion_model = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=1,
        )

        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

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
        polarity = TextBlob(sentence).sentiment.polarity

        if polarity > 0.1:
            label = "positive"
        elif polarity < -0.1:
            label = "negative"
        else:
            label = "neutral"

        return label, float(abs(polarity))

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

    def _cluster_topic_labels(self, topic_labels, max_topics=10):

        # --- Takes detailed topic labels and groups them into max 10 high-level categories.

        unique_labels = list(set(topic_labels))

        if len(unique_labels) <= 1:
            return {label: "General" for label in unique_labels}

        if len(unique_labels) <= max_topics:
            return {label: self._make_short_label(label) for label in unique_labels}

        topic_model = BERTopic(
            embedding_model=self.embedding_model,
            nr_topics=max_topics,
            min_topic_size=2,
            verbose=False,
    )

        topic_ids, _ = topic_model.fit_transform(unique_labels)

        cluster_names = {}

        for topic_id in set(topic_ids):
            if topic_id == -1:
                cluster_names[topic_id] = "General"
                continue

            topic_words = topic_model.get_topic(topic_id)

            if not topic_words:
                cluster_names[topic_id] = "General"
                continue

            top_words = [word for word, _ in topic_words[:2]]
            cluster_names[topic_id] = self._make_short_label(" ".join(top_words))

        label_to_high_level = {}

        for label, topic_id in zip(unique_labels, topic_ids):
            label_to_high_level[label] = cluster_names.get(
                topic_id, self._make_short_label(label)
        )

        return label_to_high_level

    def _make_short_label(self, text):
        """
        Creates a max-2-word topic label.
        """

        words = re.findall(r"[A-Za-z]+", text)

        stopwords = {
            "the",
            "and",
            "or",
            "but",
            "with",
            "about",
            "that",
            "this",
            "what",
            "when",
            "where",
            "why",
            "how",
            "you",
            "your",
            "they",
            "them",
            "are",
            "was",
            "were",
            "can",
            "could",
            "would",
            "should",
            "like",
            "really",
            "just",
            "something",
        }

        meaningful_words = [
            word.title()
            for word in words
            if word.lower() not in stopwords and len(word) > 2
        ]

        if not meaningful_words:
            return "General"

        return " ".join(meaningful_words[:2])

