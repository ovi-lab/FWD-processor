import json

import pandas

# Paths
PATH_TO_SAMPLE_CONVERSATION = "data/log-01_22_2024, 06_22_56 PM.json"
# Constants
NODE_SIZE_INCREMENT_BY = 0.25


def main():
    """
    Reads a file with a list of conversation and uses the DialogTag Python tool to
    predict the dialogue tag of each line of conversation and groups the conversation
    based on the dialogue tag functional groups. The information obtained are then
    organized into a dictionary and saved as a JSON file to feed into the d3.js script.

    NOTE: DialogTag required Python 3.7 or higher, Tensorflow 2.0.0 or higher
          and Transformers v3.0.0 or higher. (Recommended: run using Google Collab)
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

    def link_exists(link_id, link_section):
        """
        Iterate through the global variable mapped_conversation
        and return true if a link with the desired link_id is found,
        else false.

        Parameters
        ----------
        link_id : int
            id of link to find

        Returns
        -------
        boolean
            true if link is found
        """
        for l in link_section:
            if l["name"] == link_id:
                return True
        return False

    def get_node_idx(node_name):
        """
        Iterate through the the global variable mapped_conversation
        and return true if a node with the desired node_id is found,
        else false.

        Parameters
        ----------
        node_id : int
            id of node to find

        Returns
        -------
        int
            index of the node with the specified node_id
        """
        nodes = mapped_conversation["nodes"]
        new_id = 0
        for i in range(len(nodes)):
            if nodes[i]["name"] == node_name:
                return i
            new_id = i
        # If no match for node name, create a new node with index
        nodes.append({
                    "id": new_id,
                    "name": node_name,
                    "grp": 0,
                    "n": 5,
                    "interactions": [],
                    "show": "false",
                })
        return new_id


    def node_has_links(n, link_section):
        """
        Iterate through the global variable mapped_conversation
        and return true if a link contains the specified node,
        else false.

        Parameters
        ----------
        n : Object
            A node object with integer attribute 'id'

        Returns
        -------
        boolean
            true if link contains node n
        """
        for l in link_section:
            if n["id"] in [l["source"], l["target"]]:
                return True
        return False

    def add_conversation(link_id, conversation):
        """
        Iterate through the list of all the links and appends the conversation to
        the link with the specified link_id

        Parameters
        ----------
        link_id : int
            id of link to find
        conversation: Object
            conversation item with attributes id, thread_if, request, response, and sentiment
            (see JSON schema for more information)

        Returns
        -------
        None
        """
        for i in range(len(mapped_conversation["links"])):
            if mapped_conversation["links"][i]["id"] == link_id:
                mapped_conversation["links"][i]["interactions"].append(conversation)
            return

    
    # Read sample conversation into a Pandas dataframe
    df = pandas.read_json(PATH_TO_SAMPLE_CONVERSATION)

    # Remove empty rows
    # df = df.dropna()
    
#     # Add a new column as needed
    df["timestamp"] = '00:00:00'  # Each line of conversation belongs to a thread and each thread has an ID


    # Create the JSON template for the results to feed into d3.js
    mapped_conversation = {
        "nodes": [],
        "links": {"haru": [], "user": []},
        "attributes": {},
    }
    print(mapped_conversation)

    # Track the last speaker. Links should only occur between two different speakers
    # i.e. If the last speaker and the current speaker is the same, no link will form
    prev_turn = ""
    # Track the id of the node that is requesting a response. Each link shows a
    # relationship between a request and a response.
    prev_node_id = 0
    prev_node_name = ""

    # Track the text of the request node
    prev_sentence = ""

    # Iterate through the list of conversations and create links as needed
    for index, row in df.iterrows():
        response_intent = row["Intent"]
        response_sentence = row['Sentence']
        response_turn = row['Turn']
        response_node_name = f"{response_turn}: {response_intent}"  # ex. "HARU: Action-directive"
        response_node_id = mapped_conversation["nodes"][get_node_idx(response_node_name)]["id"]
        
        new_interaction = {
            "idx": row['idx'],
            "index": row['Index'],
            "highlighted": row['Highlighted'],
            "intent": row['Intent'],
            "prompt": prev_sentence,
            "sentence": row['Sentence'],
            "turn": row['Turn'],
            "lastInteraction": row['LastInteraction'],
            "app_name": row['app_name'],
            "emotion_label": row['emotion_label'],
            "emotion_score": row['emotion_score'],
            "sentiment_label": row['sentiment_label'],
            "sentiment_score": row['sentiment_score'],
            "timestamp": row['timestamp'],
        }
        
        # # If the same speaker speaking means new thread
        # This program assumes that a speaker will not speak more
        # than one time consecutively in the same thread
        # if response_turn == prev_turn:
        #     prev_node_name = response_node_name
        #     prev_node_id = response_node_id
        #     prev_sentence = response_sentence
        #     continue
        
        # Adding more information to each node
        mapped_conversation["nodes"][get_node_idx(response_node_name)][
            "interactions"
        ].append(new_interaction)

        # increase node size
        increase_node_size(response_node_id)

        # If this is the first node or in case of error where
        # last speaker is empty, no link will be created
        if prev_turn == "":      
            prev_node_id = response_node_id
            prev_turn = response_turn
            prev_sentence = response_sentence
            prev_node_name = response_node_name
            continue

       
        link_section = mapped_conversation["links"][new_interaction['turn']]
        link_name = f"{prev_node_name} to {response_node_name}"  # ex. "HARU: Action-directive to CHILD: Question"

        # If the link already exists, don't create another one
        if link_exists(link_name, link_section):
            for link in link_section:
                if link["name"] == link_name:
                    link["interactions"].append(new_interaction)
                    link["value"] += 1

        # If a link between the request and response node does not
        # exist yet, create a new link with value 1
        else:
            link_section.append(
                {
                    "name": link_name,
                    "source": prev_node_id,
                    "target": response_node_id,
                    "value": 1,
                    "interactions": [new_interaction],
                }
            )
            # Set the new values for the following flags
        prev_node_name = response_node_name
        prev_node_id = response_node_id
        prev_sentence = response_sentence
        prev_turn = response_turn

    for i in range(len(mapped_conversation["nodes"])):
        if node_has_links(mapped_conversation["nodes"][i], link_section):
            mapped_conversation["nodes"][i]["show"] = "true"

    # Save the results as a JSON file
    print(mapped_conversation)
    path = r"output/new_robot_data.json"
    try:
        with open(path, "x") as f:
            json.dump(mapped_conversation, f, ensure_ascii=False, indent=4)
    except FileExistsError:
        print("File exists, overwriting with new data")
        with open(path, "w") as f:
            json.dump(mapped_conversation, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
