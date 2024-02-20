import json

key_map = {
        "Highlighted": "highlighted",
        "Index": "index",
        "Intent": "intent",
        "LastInteraction": "lastinteraction",
        "Sentence": "sentence",
        "Turn": "turn",
        "app_name": "app_name",
        "emotion_label": "emotion_label",
        "emotion_score": "emotion_score",
        "idx": "idx",
        "intent_category": "intent_category",
        "sentiment_label": "sentiment_label",
        "sentiment_score": "sentiment_score",
        "timestamp": "timestamp",
        "topic_name": "topic_name"
    }

def replace_keys(original_dict):
    new_dict = {}
    for old_key, new_key in key_map.items():
        if old_key in original_dict:
            new_dict[new_key] = original_dict[old_key]
    return new_dict


# Open the JSON file
with open('data/log-02-20.json', encoding='utf8') as json_file:
    # Load JSON data
    data = json.load(json_file)

data.reverse()
# updated_list = []
# for dict_ in data :
#     new_dict = replace_keys(dict_)
#     updated_list.append(new_dict)


# Specify the file path
file_path = 'data/log-02-20.json'

# Dump data to JSON file
with open(file_path, 'w') as json_file:
    json.dump(data, json_file)