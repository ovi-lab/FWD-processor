"""rebuilds the JSON file with the values actually needed for the visualizations and reverses"""

import json
from datetime import datetime

interaction_key_map = {
    "Highlighted": "highlighted",
    "Intent": "intent",
    "Sentence": "sentence",
    "Turn": "turn",
    "app_name": "app_name",
    "emotion_label": "emotion_label",
    "emotion_score": "emotion_score",
    "entity_type_detection": "entity_type_detection",
    "idx": "idx",
    "index": "index",
    "intent_category": "intent_category",
    "lastinteraction": "last_interaction",
    "sentiment_label": "sentiment_label",
    "sentiment_score": "sentiment_score",
    "timestamp": "timestamp",
    "topic_name": "topic",
    "type": "type",
}

TOPIC_KEY_PATH = "conversation_analysis/ToF-topics/topic-intent-key.json"
topic_list = []
with open(TOPIC_KEY_PATH, "r", encoding="utf8") as topic_file:
    topic_list = json.load(topic_file)


def replace_keys(original_dict):
    """replace the keys from the original data to remove caps and rename for clarity"""
    new_dict = {}
    for old_key, new_key in interaction_key_map.items():
        if old_key in original_dict:
            new_dict[new_key] = original_dict[old_key]
    return new_dict


def generate_topic(dialog_line, last_topic_group):
    """use the intents from the data to make sure the correct topic is assigned to each line"""

    topic_name_output = ""
    if last_topic_group:
        for intent in last_topic_group["intents"]:
            if intent == dialog_line["intent"]:
                topic_name_output = last_topic_group["topic_name"]
                return topic_name_output, last_topic_group

    for topic in topic_list:
        if topic["topic_name"] in dialog_line["intent"]:
            topic_name_output = topic["topic_name"]
            return topic_name_output, topic

    for topic in topic_list:
        for intent in topic["intents"]:
            if intent == dialog_line["intent"]:
                topic_name_output = topic["topic_name"]
                return topic_name_output, topic
            
    
    print(f"TOPIC FOR *{dialog_line['intent']}* NOT FOUND \n")
    return topic_name_output, last_topic_group


def fix_index(input_list):
    """Fix the index of an original (unprocessed) log in the case that the index is cut off"""
    input_list.reverse()
    idx = 0
    for line in input_list:
        line["idx"] = idx
        idx += 1
    input_list.reverse()
    return input_list


# Specify the file paths
# INPUT_FILE_PATH = "data/log-02_20_2024, 04_27_07 PM.json"
INPUT_FILE_PATH = "output/dialog_output.json"
OUTPUT_FILE_PATH = "data/log-02-20.json"

# Open the JSON file
data = []
with open(INPUT_FILE_PATH, "r", encoding="utf8") as json_file:
    # Load JSON data
    data = json.load(json_file)


# updated_list = fix_index(data)
updated_list = []
last_topic = ""

start = datetime.now()
for utterance in data:
    updated_line = utterance
    updated_line["topic"], last_topic = generate_topic(updated_line, last_topic)
    updated_list.append(updated_line)


# Dump data to JSON file
with open(OUTPUT_FILE_PATH, "w", encoding="utf8") as json_file:
    print('writing...')
    json.dump(updated_list, json_file)
    print('done.')
    end = datetime.now()
    print(end - start)
