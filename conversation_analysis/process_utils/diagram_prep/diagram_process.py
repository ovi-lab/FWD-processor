import json
from datetime import datetime
# from itertools import pairwise # This import does not work on Python versions that are < 3.10
from itertools import tee
import process_utils.diagram_prep.diagram_utils as utils


def pairwise(iterable):
    """Returns an iterator of paired items, overlapping, from the original

    >>> take(4, pairwise(count()))
    [(0, 1), (1, 2), (2, 3), (3, 4)]

    Since the we have to run everything on Python 3.8 on the silverbox, 
    we can't use Python's itertools pairwise, so we are using the old pairwise recipe from pre Python 3.10
    - If you are now running Python >= 3.10 feel free to comment out this function and import itertools pairwise if desired
    
    On Python 3.10 and above, this is an alias for :func:`itertools.pairwise`.

    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

    
def generate_diagram_from_file(transcript_data, diagram_json_output):
    """Iterates through the transcript JSON file data and creates nodes for each topic for both speakers, 
    records ever line of dialogue uttered within that topic, and populates a list of links whenever 
    there is a topic linking to another

    Args:
        transcript_data (_type_): _description_
        diagram_json_output (String): filepath for where we want to save the diagram output

    Returns:
        _type_: _description_
    """
    
    mapped_conversation = {
        "nodes": [],
        "links": {"haru": [], "user": []},
        "attributes": {},
    }
    mapped_nodes = mapped_conversation["nodes"]
    mapped_links = mapped_conversation["links"]


    for prompt, response in pairwise(transcript_data):
        response_timestamp = str(prompt["timestamp"])
        response_generated_idx = utils.select_or_create_node(
            mapped_nodes, prompt["topic"], prompt["turn"], response_timestamp
        )
        prompt_generated_idx = utils.select_or_create_node(
            mapped_nodes, response["topic"], response["turn"], response_timestamp
        )
        response_node = mapped_nodes[response_generated_idx]
        prompt_node = mapped_nodes[prompt_generated_idx]

        new_interaction = {
            "idx": prompt["idx"],
            "index": prompt["index"],
            "response": response["sentence"],
            "sentence": prompt["sentence"],
            "turn": prompt["turn"],
            "topic": prompt["topic"],
            "intent": prompt["intent"],
            "emotion_label": prompt["emotion_label"],
            "emotion_score": float(prompt["emotion_score"]),
            "sentiment_label": prompt["sentiment_label"],
            "sentiment_score": float(prompt["sentiment_score"]),
            "last_interaction": bool(prompt["last_interaction"]),
            "data_collection": prompt["entity_type_detection"],
            "timestamp": response_timestamp,
            "slots": prompt["slots"]
        }

        # Adding more information to each node
        mapped_nodes[response_generated_idx]["interactions"].append(new_interaction)

        response_node['slots'] = utils.add_node_slots(response_node, prompt['slots'])

        mapped_nodes[response_generated_idx]["timestamp"] = str(
            prompt["timestamp"]
        )

        if response_node["name"] == prompt_node["name"]:
            continue

        link_turn_section = mapped_links[response["turn"]]
        link_name = f"{prompt_node['name']} -> {response_node['name']}"  # ex. "HARU: Action-directive to CHILD: Question"

        # If the link already exists, don't create another one
        if utils.link_exists(link_name, link_turn_section):
            for link in link_turn_section:
                if link["name"] == link_name:
                    link["timestamp"] = str(prompt["timestamp"])
                    link["responseIDs"].append(new_interaction["idx"])
                    link["value"] += 1

        # If a link between the request and prompt node does not
        # exist yet, create a new link with value 1
        else:
            link_turn_section.append(
                {
                    "name": link_name,
                    "source": prompt_node["id"],
                    "target": response_node["id"],
                    "value": 1,
                    "timestamp": str(prompt["timestamp"]),
                    "responseIDs": [new_interaction["idx"]],
                }
            )

    for node in mapped_nodes:
        node["show"] = utils.node_has_links(node, link_turn_section)

    # Save the results as a JSON file

    with open(diagram_json_output, "w", encoding="utf8") as f:
        json.dump(mapped_conversation, f, ensure_ascii=False, indent=4)

    return mapped_conversation

if __name__ == "__main__":
    # Paths
    PATH_TO_SAMPLE_CONVERSATION = "../data/transcript_data/transcript-log.json"
    OUT_PATH = r"../data/diagram_data/new_robot_data_03_12.json"

    start = datetime.now()
    generate_diagram_from_file(PATH_TO_SAMPLE_CONVERSATION, OUT_PATH)
    end = datetime.now()
    print(end - start)