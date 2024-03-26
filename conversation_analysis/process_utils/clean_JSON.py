"""rebuilds the JSON file with the values actually needed for the visualizations and reverses"""

import os
import json
from datetime import datetime

current_directory = os.path.dirname(os.path.abspath(__file__))

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

TOPIC_KEY_PATH = os.path.join(current_directory, "ToF-topics/topic-intent-key.json")
with open(TOPIC_KEY_PATH, "r", encoding="utf8") as topic_file:
    topic_list = json.load(topic_file)


def replace_keys(original_dict):
    """replace the keys from the original data to remove caps and rename for clarity"""
    new_dict = {}
    for old_key, new_key in interaction_key_map.items():
        if old_key in original_dict:
            new_dict[new_key] = original_dict[old_key]
    return new_dict


def generate_topic(dialog_line_intent):
    """use the intents from the data to make sure the correct topic is assigned to each line"""

    topic_name_output = ""
    # if last_topic_group:
    #     for intent in last_topic_group["intents"]:
    #         if intent == dialog_line_intent:
    #             topic_name_output = last_topic_group["topic_name"]
    #             return topic_name_output, last_topic_group

    for topic in topic_list:
        if topic["topic_name"] in dialog_line_intent:
            topic_name_output = topic["topic_name"]
            return topic_name_output

    for topic in topic_list:
        for intent in topic["intents"]:
            if intent == dialog_line_intent:
                topic_name_output = topic["topic_name"]
                return topic_name_output
            
    
    print(f"TOPIC FOR *{dialog_line_intent}* NOT FOUND \n")
    return topic_name_output


def fix_index(input_list):
    """Fix the index of an original (unprocessed) log in the case that the index is cut off"""
    idx = 0
    for line in input_list:
        line["idx"] = idx
        idx += 1
    input_list.reverse()    
    return input_list

def process_conversation(data, output_file=''):
    updated_list = []
    last_topic = ''

    start = datetime.now()

    for utterance in data:
        updated_line = utterance
        # updated_line["topic"], last_topic = generate_topic(updated_line['intent'], last_topic)
        updated_list.append(updated_line)
    updated_list = fix_index(updated_list)

    if output_file:
        # Dump data to JSON file
        with open(output_file, "w", encoding="utf8") as json_file:
            print('writing...')
            json.dump(updated_list, json_file, indent=4)
            print('done.')
            end = datetime.now()
            print(end - start)
    return updated_list

def process_from_file(input_file, output_file=''):
    # Open the JSON file
    with open(input_file, "r", encoding='utf8') as json_file:
        # Load JSON data
        data = json.load(json_file)
    return process_conversation(data, output_file)

if __name__ == '__main__':
    INPUT_FILE_PATH = "data/dialog_output.json"
    OUTPUT_FILE_PATH = "data/log-03-12.json"
    process_from_file(INPUT_FILE_PATH, OUTPUT_FILE_PATH)