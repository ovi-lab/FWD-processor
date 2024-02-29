from openpyxl import load_workbook
import json

FILE_PATH = 'conversation-analysis/ToF-topics/Tiers-of-Friendship.xlsx'

wb = load_workbook(FILE_PATH)
topic_list = []


def add_to_topic(current_topic, current_intent):
    if (len(topic_list) < 1):
        topic_list.append({"topic": current_topic,
                           "intents": [current_intent]})
        return
    for t in topic_list:
        if t['topic'] == current_topic:
            if (current_intent != t['intents'][-1]):
                t['intents'].append(current_intent)
            return
    topic_list.append({"topic": current_topic, "intents": [current_intent]})

for sheet in wb:
    topic = sheet['A'][1].value
    for intent in sheet['B']:
        if (intent.row == 1):
            continue
        add_to_topic(topic, intent.value)

OUTPUT_PATH = "conversation-analysis/ToF-topics/topic-intent-key.json"

with open(OUTPUT_PATH, 'w', encoding='utf8') as json_file:
    json.dump(topic_list, json_file)