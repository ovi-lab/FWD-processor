"""
Functions for converting the raw .txt ROS responses 
to a legible JSON file for further processing
the code has been modified from Lithin's dynamic ROS subscriber
"""

from time import time
from datetime import datetime
import yaml
from process_utils.clean_JSON import generate_topic


EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
GENRE_MAPPING = {
    "anger": "whiny",
    "disgust": "whiny",
    "fear": "serious",
    "joy": "highnrg",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": "highnrg",
}
SENTIMENT_MAPPING = {
    "neg": "negative",
    "neu": "neutral",
    "pos": "positive",
}


class Timer:
    def __init__(self) -> None:
        self._start_time = 0
        self._stop_time = 0

    def start_timer(self):
        self._start_time = time()

    def stop_timer(self):
        self._stop_time = time()

    def duration(self):
        return self._stop_time - self._start_time


class HaruChatObject:
    def __init__(self, data) -> None:
        self._data = data
        self.index = 0
        self.last_user_utterance = ""
        self.app_name = "smalltalk"
        self.conversation = []
        # Set Parameters
        self.data_type_list = [
            "sentence_list",
            "sentence_id",
            "sentence",
            "emotion_name",
            "emotion_score",
            "sentiment_name",
            "sentiment_score",
            "emotion_frequency",
            "probability_emote",
            "random_value",
            "react",
            "reaction_text",
            "sentence_processed",
            "sentence_processed_list",
        ]
        self.data_keys = {
            key2: {key: None for key in self.data_type_list}
            for key2 in ["user", "haru"]
        }
        self.data_keys["index"] = None
        self._lastinteraction = None

        self.output = self.parse_raw_dialog(data)

    def parse_raw_dialog(self, passed_data):
        for doc in passed_data:
            if doc is None:
                break
            self.data_keys["index"] = doc["header"]["seq"]
            self.data_keys["haru"]["sentence_list"] = doc["fulfillment_sentences"]

            self.combined_haru_emotion_label = doc["fulfillment_emotion"]["emotions"][
                "results"
            ]["best_match"]["label"]
            self.combined_haru_emotion_score = doc["fulfillment_emotion"]["emotions"][
                "results"
            ]["best_match"]["score"]
            self.combined_user_emotion_label = doc["utterance_emotion"]["emotions"][
                "results"
            ]["best_match"]["label"]
            self.combined_user_emotion_score = doc["utterance_emotion"]["emotions"][
                "results"
            ]["best_match"]["score"]
            self.slu_intent = doc["slu_result"]["intent"]
            self.topic_name = generate_topic(doc["slu_result"]["intent"])
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.generic_entity_type_detection = False
            if doc["slots"]:
                for slots in doc["slots"]:
                    if slots["name"] == "enable_intent_classification":
                        self.generic_entity_type_detection = True
                    else:
                        self.generic_entity_type_detection = False

            new_dialog_node = {
                "idx": self.index, # const
                "sentence": doc["utterance"],
                "emotion_label": "neutral", # var with default value
                "emotion_score": -9, # var with default value
                "sentiment_label": "neutral", # var with default value
                "sentiment_score": -9, # var with default value
                "index": doc["header"]["seq"], # const
                "app_name": self.app_name, # const
                "turn": "user", # var with default value
                "last_interaction": False, # default value
                "highlighted": False, # var with default value
                "intent": self.slu_intent, # const
                "topic": self.topic_name,  # const
                "timestamp": self.timestamp, # const
                "type": "", # var
                "entity_type_detection": self.generic_entity_type_detection, # const
                "slots": doc["slots"], # const
            }
            
            if self.last_user_utterance != doc["utterance"]:
                self.last_user_utterance = doc["utterance"]

                if len(doc["utterance_sentences"]) == 0:
                    if self._lastinteraction is not None and self.conversation:
                        self.conversation[self._lastinteraction][
                            "lastinteraction"
                        ] = False
                    if self.data_keys["user"]["sentence_list"] is not None:
                        self.conversation.append(
                            {
                                "idx": self.index,
                                "sentence": doc["utterance"],
                                "emotion_label": "neutral",
                                "emotion_score": -9,
                                "sentiment_label": "neutral",
                                "sentiment_score": -9,
                                "index": doc["header"]["seq"],
                                "app_name": self.app_name,
                                "turn": "user",
                                "last_interaction": True,
                                "highlighted": False,
                                "intent": self.slu_intent,
                                "topic": self.topic_name,
                                "timestamp": self.timestamp,
                                "type": "",
                                "entity_type_detection": self.generic_entity_type_detection,
                                "slots": doc["slots"],
                            }
                        )
                    self.index += 1
                    self._lastinteraction = 0
                else:

                    self.data_keys["user"]["sentence_list"] = doc["utterance_sentences"]
                    if self._lastinteraction is not None and self.conversation:
                        self.conversation[self._lastinteraction][
                            "lastinteraction"
                        ] = False
                    if self.data_keys["user"]["sentence_list"] is not None:
                        for sentence_id, user in enumerate(
                            self.data_keys["user"]["sentence_list"]
                        ):
                            try:
                                user_emotion_label = user["emotion_results"][
                                    "emotions"
                                ]["results"]["best_match"]["label"]
                                user_emotion_score = user["emotion_results"][
                                    "emotions"
                                ]["results"]["best_match"]["score"]
                            except Exception:
                                if user["emotion"] != "":
                                    user_emotion_label = user["emotion"]
                                    user_emotion_score = 1.0
                                elif user["auto_emotion"] != "":
                                    user_emotion_label = user["auto_emotion"]
                                    user_emotion_score = (
                                        float(user["auto_score"])
                                        if user["auto_score"] != ""
                                        else 0.0
                                    )
                                else:
                                    user_emotion_label = (
                                        self.combined_user_emotion_label
                                    )
                                    user_emotion_score = (
                                        self.combined_user_emotion_score
                                    )

                            sentiment_score = user["sentiment_results"]["sentiment"][
                                "results"
                            ]["best_match"]["score"]
                            sentiment_label = SENTIMENT_MAPPING.get(
                                user["sentiment_results"]["sentiment"]["results"][
                                    "best_match"
                                ]["label"],
                                "None",
                            )
                            self.conversation.append(
                                {
                                    "idx": self.index,
                                    "sentence": user["text"],
                                    "emotion_label": user_emotion_label,
                                    "emotion_score": round(user_emotion_score, 2),
                                    "sentiment_label": sentiment_label,
                                    "sentiment_score": round(sentiment_score, 2),
                                    "index": doc["header"]["seq"],
                                    "app_name": self.app_name,
                                    "turn": "user",
                                    "last_interaction": sentence_id == 0,
                                    "highlighted": False,
                                    "intent": self.slu_intent,
                                    "topic": self.topic_name,
                                    "timestamp": self.timestamp,
                                    "type": user["type"],
                                    "entity_type_detection": self.generic_entity_type_detection,
                                    "slots": doc["slots"],
                                }
                            )
                            self.index += 1
                            if sentence_id == 0:
                                self._lastinteraction = len(self.conversation) - 1
            if self.data_keys["haru"]["sentence_list"] is not None:
                for sentence_id, haru in enumerate(
                    self.data_keys["haru"]["sentence_list"]
                ):
                    if "|" in haru["text"]:
                        continue
                    try:
                        haru_emotion_label = haru["emotion_results"]["emotions"][
                            "results"
                        ]["best_match"]["label"]
                        haru_emotion_score = haru["emotion_results"]["emotions"][
                            "results"
                        ]["best_match"]["score"]
                    except Exception:
                        if haru["emotion"] != "":
                            haru_emotion_label = haru["emotion"]
                            haru_emotion_score = 1.0
                        elif haru["auto_emotion"] != "":
                            haru_emotion_label = haru["auto_emotion"]
                            haru_emotion_score = (
                                float(haru["auto_score"])
                                if haru["auto_score"] != ""
                                else 0.0
                            )
                        else:
                            haru_emotion_label = self.combined_haru_emotion_label
                            haru_emotion_score = self.combined_haru_emotion_score
                    sentiment_score = haru["sentiment_results"]["sentiment"]["results"][
                        "best_match"
                    ]["score"]
                    sentiment_label = SENTIMENT_MAPPING.get(
                        haru["sentiment_results"]["sentiment"]["results"]["best_match"][
                            "label"
                        ],
                        "None",
                    )
                    self.conversation.append(
                        {
                            "idx": self.index,
                            "sentence": haru["text"],
                            "emotion_label": haru_emotion_label,
                            "emotion_score": round(haru_emotion_score, 2),
                            "sentiment_label": sentiment_label,
                            "sentiment_score": round(sentiment_score, 2),
                            "index": doc["header"]["seq"],
                            "app_name": self.app_name,
                            "turn": "haru",
                            "last_interaction": False,
                            "highlighted": False,
                            "intent": self.slu_intent,
                            "topic": self.topic_name,
                            "timestamp": self.timestamp,
                            "type": haru["type"],
                            "entity_type_detection": self.generic_entity_type_detection,
                            "slots": doc["slots"],
                        }
                    )
                    self.index += 1
            self._data_is_ready = True

        return self.conversation
    
def parse_yml_output(ros_yml_path):
    try:
        with open(ros_yml_path, "r", encoding="utf8") as file:
            print(f'converting the static YML file at "{ros_yml_path}" to JSON...')
            data = yaml.safe_load_all(file)
            haru_chat = HaruChatObject(data)
            return haru_chat.output

    except FileNotFoundError as e:
        print(e)
        return False

if __name__ == "__main__":
    YML_PATH = "data/dialog_result.yml"
    OUTPUT_PATH = "output/dialog_output.json"
    parse_yml_output(YML_PATH)
