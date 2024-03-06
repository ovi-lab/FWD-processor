import json
from datetime import datetime
from itertools import pairwise
import ijson
import diagram_utils as utils

# Paths
PATH_TO_SAMPLE_CONVERSATION = "data/log-02-20.json"

PATH_TO_ONTOLOGY = 'data/Tiers-of-Friendship.xlsx'



def diagram_process():
    """
    Reads a file with a list of conversation and uses the DialogTag Python tool to
    predict the dialogue tag of each line of conversation and groups the conversation
    based on the dialogue tag functional groups. The information obtained are then
    organized into a dictionary and saved as a JSON file to feed into the d3.js script.
    """

    # Read sample conversation into a Pandas dataframe
    # df = pandas.read_json(PATH_TO_SAMPLE_CONVERSATION, encoding="utf8")
    # row_iterator = df.iterrows() # pylint: disable=E1101

    # Create the JSON template for the results to feed into d3.js
    mapped_conversation = {
        "nodes": [],
        "links": {"haru": [], "user": []},
        "attributes": {},
    }
    mapped_nodes = mapped_conversation["nodes"]
    mapped_links = mapped_conversation["links"]

    with open(PATH_TO_SAMPLE_CONVERSATION, 'rb') as f:
        row_iterator = ijson.items(f, 'item')

        for response, prompt in pairwise(row_iterator):
            response_timestamp = str(response["timestamp"])
            response_generated_idx = utils.select_node(mapped_nodes, response["topic"], response["turn"], response_timestamp)
            prompt_generated_idx = utils.select_node(mapped_nodes, prompt["topic"], prompt["turn"], response_timestamp)
            response_node = mapped_nodes[response_generated_idx]
            prompt_node = mapped_nodes[prompt_generated_idx]

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
                "emotion_score": float(response["emotion_score"]),
                "sentiment_label": response["sentiment_label"],
                "sentiment_score": float(response["sentiment_score"]),
                "last_interaction": bool(response["last_interaction"]),
                "data_collection": response["entity_type_detection"],
                "timestamp": response_timestamp,
            }

            # Adding more information to each node
            mapped_nodes[response_generated_idx][
                "interactions"
            ].append(new_interaction)

            mapped_nodes[response_generated_idx]["timestamp"] = str(response['timestamp'])

            if (response_node['name'] == prompt_node['name']):
                continue

            link_turn_section = mapped_links[prompt["turn"]]
            link_name = f"{prompt_node['name']} -> {response_node['name']}"  # ex. "HARU: Action-directive to CHILD: Question"

            # If the link already exists, don't create another one
            if utils.link_exists(link_name, link_turn_section):
                for link in link_turn_section:
                    if link["name"] == link_name:
                        link["timestamp"] = str(response["timestamp"])
                        link["responseIDs"].append(new_interaction["idx"])
                        link["value"] += 1

            # If a link between the request and response node does not
            # exist yet, create a new link with value 1
            else:
                link_turn_section.append(
                    {
                        "name": link_name,
                        "source": prompt_node["id"],
                        "target": response_node["id"],
                        "value": 1,
                        "timestamp": str(response["timestamp"]),
                        "responseIDs": [new_interaction["idx"]],
                    }
                )

    for node in mapped_nodes:
            node["show"] = utils.node_has_links(node, link_turn_section)

    # Save the results as a JSON file
    path = r"output/new_robot_data_2_20.json"
    try:
        with open(path, "x", encoding="utf8") as f:
            json.dump(mapped_conversation, f, ensure_ascii=False, indent=4)
    except FileExistsError:
        print("File exists, overwriting with new data")
        with open(path, "w", encoding="utf8") as f:
            json.dump(mapped_conversation, f, ensure_ascii=False, indent=4)
    print('done')


if __name__ == "__main__":
    start = datetime.now()
    diagram_process()
    end = datetime.now()
    print(end-start)
