"""
Optional local-LLM (Ollama) passes that run alongside the emotion model and VADER:

1. correct_transcription: lightly fixes obvious speech-to-text mistakes in the child's lines
   ("Mass" -> "Math" after "What's your favourite subject?"). Corrections that change too
   much are rejected, so the child's own wording is kept.
2. review_labels: double-checks each line's emotion and sentiment label and changes only
   ones that are clearly wrong, choosing from the same label sets the models use.

Both send lines in batches with a JSON schema (Ollama structured outputs), so a transcript
takes a handful of requests. Any failure leaves the input unchanged.
"""

import difflib
import json
import re

import requests

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
SENTIMENTS = ["positive", "neutral", "negative"]


class LLMReviewer:
    BATCH_SIZE = 10
    # A correction may change at most this many words (or 20% of a long reply), and may not
    # add or remove more than one word. "Mass" -> "Math" changes 1; rewrites change many.
    MAX_CHANGED_WORDS = 2

    def __init__(self, host, model, keep_alive="30m", timeout=120):
        self.host = host
        self.model = model
        self.keep_alive = keep_alive
        self.timeout = timeout
        self.available = True

    def reset(self):
        """Call once per transcript: retry Ollama even if it failed on the last one."""
        self.available = True

    # ------------------------------------------------------------------ corrections

    def correct_transcription(self, replies):
        """
        replies: list of (robot_question, child_reply). Returns a list of the same length
        holding the corrected reply, or None where the original should be kept.
        """
        results = [None] * len(replies)

        for start in range(0, len(replies), self.BATCH_SIZE):
            batch = replies[start:start + self.BATCH_SIZE]
            # Structured input keeps the robot's question separate from the reply, so the
            # model doesn't blend the two together in its answer.
            listing = json.dumps(
                [{"id": i, "robot_question": q, "child_reply": r} for i, (q, r) in enumerate(batch)],
                ensure_ascii=False,
                indent=1,
            )
            prompt = (
                "Below are a child's replies to a social robot, written down by speech-to-text "
                "software. Find replies with an obvious speech-to-text mistake: a misheard word "
                "that makes no sense given the robot's question (for example child_reply 'Mass' "
                "when the robot asked about school subjects, meaning 'Math').\n"
                "Rules:\n"
                "- Only fix misheard words. Keep the child's own words, grammar, filler words "
                "(um, uh) and meaning.\n"
                "- Never rephrase, complete, tidy up, or add anything.\n"
                "- Return ONLY the replies you fixed, as the corrected child_reply. Most replies "
                "are fine; if none need fixing, return an empty list.\n\n"
                f"{listing}"
            )
            answer = self._generate_json(prompt, _schema({
                "id": {"type": "integer"},
                "child_reply": {"type": "string"},
            }))
            for item in _items(answer):
                i = item.get("id")
                if isinstance(i, int) and 0 <= i < len(batch):
                    question, original = batch[i]
                    results[start + i] = self._accept_correction(
                        original, item.get("child_reply"), question
                    )

        return results

    def _accept_correction(self, original, corrected, question=""):
        """Returns the correction if it's a light fix, otherwise None (keep the original)."""
        if not isinstance(corrected, str) or not corrected.strip():
            return None
        corrected = corrected.strip()

        # Only case or punctuation changed: not worth recording as a correction.
        if _normalise(corrected) == _normalise(original):
            return None

        # The model copied the prompt or the robot's line into the reply.
        if "robot" in corrected.lower() and "robot" not in original.lower():
            return None
        if question and _normalise(question)[:4] == _normalise(corrected)[:4]:
            return None

        # Light fix = a few misheard words swapped for similar-sounding ones. Nothing deleted
        # (the child's filler words stay), nothing added, nothing replaced by a different word.
        original_words, corrected_words = _normalise(original), _normalise(corrected)
        matcher = difflib.SequenceMatcher(None, original_words, corrected_words)
        changed = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            if tag != "replace":
                return None
            before = " ".join(original_words[i1:i2])
            after = " ".join(corrected_words[j1:j2])
            if not _sounds_alike(before, after):
                return None
            changed += max(i2 - i1, j2 - j1)
        if changed > max(self.MAX_CHANGED_WORDS, int(len(original_words) * 0.2)):
            return None

        return corrected

    # ------------------------------------------------------------------ labels

    def review_labels(self, lines):
        """
        lines: list of dicts with speaker, text, emotion, sentiment.
        Returns a list of (emotion, sentiment) of the same length; unchanged where the
        model agreed, gave an invalid label, or couldn't be reached.
        """
        results = [(line["emotion"], line["sentiment"]) for line in lines]

        for start in range(0, len(lines), self.BATCH_SIZE):
            batch = lines[start:start + self.BATCH_SIZE]
            listing = json.dumps(
                [
                    {"id": i, "speaker": l["speaker"], "text": l["text"],
                     "emotion": l["emotion"], "sentiment": l["sentiment"]}
                    for i, l in enumerate(batch)
                ],
                ensure_ascii=False,
                indent=1,
            )
            prompt = (
                "You are checking emotion and sentiment labels in a conversation between a social "
                "robot and a child. Automatic models produced the labels below.\n"
                f"Emotion must be one of: {', '.join(EMOTIONS)}.\n"
                f"Sentiment must be one of: {', '.join(SENTIMENTS)}.\n"
                "Return ONLY lines whose label is clearly wrong for what was said, with the "
                "corrected emotion and sentiment. When unsure, leave the line out. If all labels "
                "are reasonable, return an empty list.\n\n"
                f"{listing}"
            )
            answer = self._generate_json(prompt, _schema({
                "id": {"type": "integer"},
                "emotion": {"type": "string", "enum": EMOTIONS},
                "sentiment": {"type": "string", "enum": SENTIMENTS},
            }))
            for item in _items(answer):
                i = item.get("id")
                if isinstance(i, int) and 0 <= i < len(batch):
                    emotion, sentiment = results[start + i]
                    if item.get("emotion") in EMOTIONS:
                        emotion = item["emotion"]
                    if item.get("sentiment") in SENTIMENTS:
                        sentiment = item["sentiment"]
                    results[start + i] = (emotion, sentiment)

        return results

    # ------------------------------------------------------------------ Ollama

    def _generate_json(self, prompt, schema):
        if not self.available:
            return None
        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "think": False,
                    "format": schema,
                    "keep_alive": self.keep_alive,
                    "options": {"temperature": 0, "num_predict": 2048},
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            return json.loads(response.json().get("response", ""))
        except Exception as e:
            print(f"[LLM review skipped] {self.model} @ {self.host}: {e}")
            self.available = False
            return None


def _schema(item_properties):
    return {
        "type": "object",
        "properties": {
            "lines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": item_properties,
                    "required": list(item_properties),
                },
            }
        },
        "required": ["lines"],
    }


def _items(answer):
    if isinstance(answer, dict) and isinstance(answer.get("lines"), list):
        return [item for item in answer["lines"] if isinstance(item, dict)]
    return []


def _one_line(text):
    return " ".join(str(text).split()).replace('"', "'")


def _normalise(text):
    return re.sub(r"[^\w\s]", "", text).lower().split()


def _sounds_alike(before, after):
    """
    Rough check that a replaced chunk could be a mishearing: similar spelling and length.
    "mass"/"math" and "waken"/"raiken" pass; "sushi"/"raiken" and "yes"/"yesterday" don't.
    """
    shorter, longer = sorted((len(before), len(after)))
    if shorter == 0 or longer / shorter > 1.8:
        return False
    return difflib.SequenceMatcher(None, before, after).ratio() >= 0.5
