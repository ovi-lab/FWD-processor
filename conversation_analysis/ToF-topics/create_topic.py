"""
Creates a new topic object for the topic list, 
assigns additional data such as tiers and variables to each new topic
currently hardcoded, but if I can get access to the backend and extract these variables 
it would work much more seemlessly
"""
topic_vars = [
    {"topic": "topic-intro", "tier": 0, "required": None, "canCollect": {"$Name": ""}},
    {
        "topic": "topic-sports",
        "tier": 1,
        "required": [{"$Name": ""}],
        "canCollect": [{"$FavSport": ""}],
    },
    {
        "topic": "topic-pets",
        "tier": 1,
        "required": [{"$Name": ""}],
        "canCollect": [{"$HasPet": ""}, {"$FavAnimal": ""}],
    },
    {
        "topic": "topic-rollercoasters",
        "tier": 1,
        "required": [{"$Name": ""}],
        "canCollect": [{"$RollerCoasterName": ""}],
    },
    {
        "topic": "topic-olympics",
        "tier": 2,
        "required": [{"$FavSport": ""}],
        "canCollect": [{"$Height": ""}],
    },
    {
        "topic": "topic-lemurs",
        "tier": 2,
        "required": [{"$HasPet": ""}, {"$FavAnimal": ""}],
        "canCollect": [{"$HomeCountry": ""}, {"$PetName": ""}],
    },
]


def create_topic(topic_name, first_intent):
    """
    Creates a new topic object and populates with first intent and tier/variable data
    """
    topic_object = {
        "topic_name": topic_name,
        "tier": 0,
        "required": [],
        "canCollect": [],
        "intents": [first_intent],
    }
    for t_info in topic_vars:
        if t_info["topic"] == topic_name:
            topic_object['tier'] = t_info['tier']
            topic_object['required'] = t_info['required']
            topic_object['canCollect'] = t_info['canCollect']
    return topic_object
