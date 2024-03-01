import json
from itertools import pairwise
import utils
import pandas

# Paths
PATH_TO_SAMPLE_CONVERSATION = "data/log-02-20.json"

PATH_TO_TOPIC_KEY = './ToF-topics/topic-intent-key.json'

PATH_TO_OUTPUT = 'output/TOF_data_2_20.json'

# Constants
NODE_SIZE_INCREMENT_BY = 0.25


def main():
    """
    Reads the conversation log and creates a list of nodes similar to that of the diagrams, 
    but focuses on the data collected in an interaction and the prerequisite data needed 
    to traverse that conversation topic. 
    """

    # Read sample conversation into a Pandas dataframe
    df = pandas.read_json(PATH_TO_SAMPLE_CONVERSATION, encoding="utf8")

    # Create the JSON template for the results to feed into d3.js
    mapped_nodes = []

    row_iterator = df.iterrows() # pylint: disable=E1101

    def create_node_name(line):
        return f'{line['turn']}: {line['topic']}'

    for response_tuple, prompt_tuple in pairwise(row_iterator):
        response = response_tuple[1]
        prompt = prompt_tuple[1]
        response_timestamp = str(response["timestamp"])
        response_node_name = create_node_name(response)
        prompt_node_name = create_node_name(prompt)
        response_generated_idx = utils.get_node_idx(mapped_nodes, response["topic"], response["turn"], response_timestamp)
        
        response_node = mapped_nodes[response_generated_idx]

        new_interaction = {
            "idx": response["idx"],
            "index": response["index"],
            "prompt": prompt["sentence"],
            "sentence": response["sentence"],
            "turn": response["turn"],
            "topic": response['topic'],
            "intent_category": response["intent_category"],
            "intent": response["intent"],
            "emotion_label": response["emotion_label"],
            "emotion_score": response["emotion_score"],
            "sentiment_label": response["sentiment_label"],
            "sentiment_score": response["sentiment_score"],
            "highlighted": response["highlighted"],
            "last_interaction": bool(response["last_interaction"]),
            "data_collection": response["entity_type_detection"],
            "timestamp": response_timestamp,
        }

        # Adding more information to each node
        mapped_nodes[response_generated_idx][
            "interactions"
        ].append(new_interaction)

        mapped_nodes[response_generated_idx]["timestamp"] = str(response['timestamp'])

        # increase node size
        response_node["n"] = utils.increase_node_size(mapped_nodes, response_node["id"], NODE_SIZE_INCREMENT_BY)

        if (response_node_name == prompt_node_name):
            continue

    # Save the results as a JSON file
    try:
        with open(PATH_TO_OUTPUT, "x", encoding="utf8") as f:
            json.dump(mapped_nodes, f, ensure_ascii=False, indent=4)
    except FileExistsError:
        print("File exists, overwriting with new data")
        with open(PATH_TO_OUTPUT, "w", encoding="utf8") as f:
            json.dump(mapped_nodes, f, ensure_ascii=False, indent=4)
    print('done')


if __name__ == "__main__":
    main()
