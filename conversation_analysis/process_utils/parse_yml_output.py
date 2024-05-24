"""
Functions for converting the raw .txt ROS responses 
to a legible JSON file for further processing
"""

from time import time
from datetime import datetime
import yaml
from conversation_analysis.process_utils.parse_subscriber_data import parse_subscriber_data
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
        self.prev_topic = ''
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
        self.conversation = parse_subscriber_data(self, doc)

    
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
