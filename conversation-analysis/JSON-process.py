import json
import utils
from itertools import pairwise
import pandas

# Paths
PATH_TO_SAMPLE_CONVERSATION = "data/log-02-20.json"

PATH_TO_ONTOLOGY = 'data/Tiers-of-Friendship.xlsx'
# Constants
NODE_SIZE_INCREMENT_BY = 0.25


def main():
    """
    Reads a file with a list of conversation and uses the DialogTag Python tool to
    predict the dialogue tag of each line of conversation and groups the conversation
    based on the dialogue tag functional groups. The information obtained are then
    organized into a dictionary and saved as a JSON file to feed into the d3.js script.
    """

    def increase_node_size(node_id):
        """
        Iterate through global variable mapped_conversation to find the
        node with the desired node_id and increase the node size.

        Parameters
        ----------
        node_id : int
            id of node to enlarge
        """
        for n in mapped_conversation["nodes"]:
            if n["id"] == node_id:
                n["n"] = n["n"] + NODE_SIZE_INCREMENT_BY

    

    def get_node_idx(node_name):
        """
        Iterate through the the global variable mapped_conversation to find the specified node
        and return the index if it is found. If not, create a new node with the name specified and return the index of the new node. 

        Parameters
        ----------
        node_name : str
            node to find

        Returns
        -------
        int
            index of the node with the specified node_name
        """
        nodes = mapped_conversation["nodes"]
        new_id = 0
        for i, node in enumerate(nodes):
            if node["name"] == node_name:
                return i
            new_id = i+1

        # If no match for node name, create a new node with index
        nodes.append(
            {
                "id": new_id,
                "name": node_name,
                "grp": 0,
                "n": 5,
                "interactions": [],
                "timestamp": str(response["timestamp"]),
                "show": "false",
            }
        )
        return new_id


    # Read sample conversation into a Pandas dataframe
    df = pandas.read_json(PATH_TO_SAMPLE_CONVERSATION, encoding="utf8")

    # Create the JSON template for the results to feed into d3.js
    mapped_conversation = {
        "nodes": [],
        "links": {"haru": [], "user": []},
        "attributes": {},
    }

    row_iterator = df.iterrows() # pylint: disable=E1101

    def create_node_name(line):
        return f'{line['Turn']}: {line['topic_name']}'

    for response_tuple, prompt_tuple in pairwise(row_iterator):
        response = response_tuple[1]
        prompt = prompt_tuple[1]
        response_node_name = create_node_name(response)
        prompt_node_name = create_node_name(prompt)
        response_node_id = mapped_conversation["nodes"][get_node_idx(response_node_name)]["id"]
        prompt_node_id = mapped_conversation["nodes"][get_node_idx(prompt_node_name)]["id"]

        

        new_interaction = {
            "idx": response["idx"],
            "index": response["index"],
            "prompt": prompt["Sentence"],
            "sentence": response["Sentence"],
            "turn": response["Turn"],
            "topic": response['topic_name'],
            "intent_category": response["intent_category"],
            "intent": response["Intent"],
            "emotion_label": response["emotion_label"],
            "emotion_score": response["emotion_score"],
            "sentiment_label": response["sentiment_label"],
            "sentiment_score": response["sentiment_score"],
            "highlighted": response["Highlighted"],
            "lastInteraction": response["lastinteraction"],
            "app_name": response["app_name"],
            "timestamp": str(response["timestamp"]),
        }

        # Adding more information to each node
        mapped_conversation["nodes"][get_node_idx(response_node_name)][
            "interactions"
        ].append(new_interaction)

        # increase node size
        increase_node_size(response_node_id)

        if (response_node_name == prompt_node_name):
            continue

        link_section = mapped_conversation["links"][prompt["Turn"]]
        print(link_section)
        link_name = f"{prompt_node_name} -> {response_node_name}"  # ex. "HARU: Action-directive to CHILD: Question"

        # If the link already exists, don't create another one
        if utils.link_exists(link_name, link_section):
            for link in link_section:
                if link["name"] == link_name:
                    link["timestamp"] = str(response["timestamp"])
                    link["responseIDs"].append(new_interaction["idx"])
                    link["value"] += 1

        # If a link between the request and response node does not
        # exist yet, create a new link with value 1
        else:
            link_section.append(
                {
                    "name": link_name,
                    "source": prompt_node_id,
                    "target": response_node_id,
                    "value": 1,
                    "timestamp": str(response["timestamp"]),
                    "responseIDs": [new_interaction["idx"]],
                }
            )

    for i in range(len(mapped_conversation["nodes"])):
        if utils.node_has_links(mapped_conversation["nodes"][i], link_section):
            mapped_conversation["nodes"][i]["show"] = "true"

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
    main()
