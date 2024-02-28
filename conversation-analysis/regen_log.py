"""rebuilds the JSON file with the values actually needed for the visualizations and reverses"""

import json

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
    "lastinteraction": "lastInteraction",
    "sentiment_label": "sentiment_label", 
    "sentiment_score": "sentiment_score", 
    "timestamp": "timestamp", 
    "topic_name": "topic", 
    "type": "type",
}

topic_list = [
    "age",
    "astronauts",
    "band",
    "birthday",
    "books",
    "clothing",
    "dinosaurs",
    "fears",
    "food",
    "hobbies",
    "hometown",
    "language",
    "lemurs",
    "marriage",
    "movies",
    "weather",
    "music",
    "olympics",
    "parents",
    "pet",
    "profession",
    "rollercoasters",
    "sports",
    "travel-homecountry",
    "travel",
    "names-origins",
    "conversation-end",
    "generic-yes-no",
    "good-bye",
    "no-action",
    "command-nudge",
    "nudge-to-speak",
    "resume-conversation-no-name",
    "resume-conversation-with-name",
    "robot-cancel",
    "thank-you",
    "game-would-you-rather",
    "humor-protocol",
    "intro",
    "generic-transition",
    "prompt",
    "unknown-language",
    "trivia-protocol",
]


def replace_keys(original_dict):
    new_dict = {}
    for old_key, new_key in interaction_key_map.items():
        if old_key in original_dict:
            new_dict[new_key] = original_dict[old_key]
    return new_dict


def generate_topic(dialog_line):
    for topic in topic_list:
        if topic in dialog_line["intent"]:
            return topic
        
def fix_index(line, idx):
    newline = line
    newline["idx"] = idx
    return newline


# Open the JSON file
with open("data/log-02-20.json", encoding="utf8") as json_file:
    # Load JSON data
    data = json.load(json_file)


updated_list = []
idx = 0
for line in data:
    # updated_line = replace_keys(line)
    updated_line = fix_index(line, idx)
    # updated_line["topic"] = generate_topic(updated_line)
    updated_list.append(updated_line)
    idx += 1

    
updated_list.reverse()

# Specify the file path
FILE_PATH = "data/log-02-20.json"

# Dump data to JSON file
with open(FILE_PATH, "w", encoding="utf8") as json_file:
    json.dump(updated_list, json_file)
