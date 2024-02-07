import pandas
import json

df = pandas.read_json("data/log-01_22_2024, 06_22_56 PM.json")

new_json_structure = []

for index, row in df.iterrows():
    new_row = {
            "idx": row['idx'],
            "index": row['Index'],
            "sentence": row['Sentence'],
            "turn": row['Turn'],
            "intent_category": "parent intent",
            "intent": row['Intent'],
            "emotion_label": row['emotion_label'],
            "emotion_score": row['emotion_score'],
            "sentiment_label": row['sentiment_label'],
            "sentiment_score": row['sentiment_score'],
            "highlighted": row['Highlighted'],
            "lastInteraction": row['LastInteraction'],
            "app_name": row['app_name'],
            "timestamp": "0000-00-00 00:00:00",
        }
    
    if (row['idx'] == 1054):
        print(row)
    
    new_json_structure.append(new_row)

path = r"data/new_log_format.json"
try:
    with open(path, "x") as f:
        json.dump(new_json_structure, f, ensure_ascii=False, indent=4)
except FileExistsError:
    print("File exists, overwriting with new data")
    with open(path, "w") as f:
        json.dump(new_json_structure, f, ensure_ascii=False, indent=4)
    