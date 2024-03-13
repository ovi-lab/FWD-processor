"""Module with helper functions for JSON-processor to use"""


def link_exists(link_id, link_section):
    """
    Iterate through the global variable mapped_conversation
    and return true if a link with the desired link_id is found,
    else false.

    Parameters
    ----------
    link_name : str
        name of link to find

    link_section : str
        turn section of links

    Returns
    -------
    boolean
        true if link is found
    """
    for l in link_section:
        if l["name"] == link_id:
            return True
    return False


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
        if n['id'] == 0:
            return False
        if n["id"] == l["source"] or n['id'] == l["target"]:
            return True
    return False


def select_or_create_node(existing_nodes, node_topic, node_turn, timestamp):
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
    new_id = 0
    node_name = f"{node_turn}: {node_topic}"
    for index, node in enumerate(existing_nodes):
        if node["name"] == node_name:
            node["n"] += 0.25
            return index
        new_id = index + 1

    # If no match for node name, create a new node with index
    existing_nodes.append(
        {
            "id": new_id,
            "name": node_name,
            "turn": node_turn,
            "topic": node_topic,
            "n": 5,
            "interactions": [],
            "timestamp": timestamp,
            "slots": [],
            "show": False,
        }
    )
    return new_id

def add_node_slots(node, response_slots):
    node_slots = node['slots']
    for slot in response_slots:
        if slot not in node_slots and slot['value']:
            node_slots.append(slot)
    return node_slots