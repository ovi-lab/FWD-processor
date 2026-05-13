"""rebuilds the JSON file with the values actually needed for the visualizations and reverses"""

import os
import json
from datetime import datetime

current_directory = os.path.dirname(os.path.abspath(__file__))

TOPIC_KEY_PATH = os.path.join(current_directory, "ToF-topics/topic-intent-key.json")
with open(TOPIC_KEY_PATH, "r", encoding="utf8") as topic_file:
    topic_list = json.load(topic_file)


def generate_topic(dialog_line_intent, prev_topic):
    """use the intents from the data to make sure the correct topic is assigned to each line"""

    topic_name_output = ""

    if dialog_line_intent == "":
        topic_name_output = "error"

    if (dialog_line_intent.__contains__("system-generic-topic-transition")):
        return prev_topic

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
    # input_list.reverse()
    return input_list

def process_conversation(data, output_file=''):
    updated_list = []

    start = datetime.now()

    for utterance in data:
        updated_line = utterance
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

def process_from_json(input_json, output_file=''):
    prev_topic = ''
    new_data = []
    input_json.reverse()
    for line in input_json:
        # line['topic'] = prev_topic = generate_topic(line['intent'], prev_topic) 
        new_data.insert(0, line)
        # new_data.append(line)
    if output_file:
        # Dump data to JSON file
        with open(output_file, "w", encoding="utf8") as json_file:
            print('writing...')
            json.dump(new_data, json_file, indent=4)
            print('done.')
    return new_data

if __name__ == '__main__':
    INPUT_FILE_PATH = "data/dialog_output.json"
    OUTPUT_FILE_PATH = "data/log-03-12.json"
    process_from_json(INPUT_FILE_PATH, OUTPUT_FILE_PATH)