"""Extracts the topics and intents from the excel repository to JSON"""

from openpyxl import load_workbook
import json
from create_topic import create_topic

FILE_PATH = "conversation-analysis/ToF-topics/Tiers-of-Friendship.xlsx"

wb = load_workbook(FILE_PATH)
topic_list = []


def populate_topic(current_topic, current_intent):
    """adds the topic information to the topic list"""
    if len(topic_list) > 0:
        for t in topic_list:
            if t["topic_name"] == current_topic:
                if current_intent != t["intents"][-1]:
                    t["intents"].append(current_intent)
                return
    topic_list.append(create_topic(current_topic, current_intent))


for sheet in wb:
    topic = sheet["A"][1].value
    for intent in sheet["B"]:
        if intent.row == 1:
            continue
        populate_topic(topic, intent.value)

OUTPUT_PATH = "conversation-analysis/ToF-topics/topic-intent-key.json"

with open(OUTPUT_PATH, "w", encoding="utf8") as json_file:
    json.dump(topic_list, json_file)
