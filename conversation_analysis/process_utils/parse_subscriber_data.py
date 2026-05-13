from datetime import datetime
from process_utils.clean_JSON import generate_topic


def parse_subscriber_data(self, data):
    self.data_keys["index"] = data["header"]["seq"]
    self.data_keys["haru"]["sentence_list"] = data["fulfillment_sentences"]

    self.combined_haru_emotion_label = data["fulfillment_emotion"]["emotions"][
        "results"
    ]["best_match"]["label"]
    self.combined_haru_emotion_score = data["fulfillment_emotion"]["emotions"][
        "results"
    ]["best_match"]["score"]
    self.combined_user_emotion_label = data["utterance_emotion"]["emotions"][
        "results"
    ]["best_match"]["label"]
    self.combined_user_emotion_score = data["utterance_emotion"]["emotions"][
        "results"
    ]["best_match"]["score"]
    self.slu_intent = data["slu_result"]["intent"]
    self.topic_name = self.prev_topic = generate_topic(data["slu_result"]["intent"], self.prev_topic)
    print('Checking...')
    print(self.topic_name)
    print(self.prev_topic)
    self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    self.generic_entity_type_detection = False
    if data["slots"]:
        for slots in data["slots"]:
            if slots["name"] == "enable_intent_classification":
                self.generic_entity_type_detection = True
            else:
                self.generic_entity_type_detection = False

    new_dialog_node = {
        "idx": self.index, # const
        "sentence": data["utterance"],
        "emotion_label": "neutral", # var with default value
        "emotion_score": -9, # var with default value
        "sentiment_label": "neutral", # var with default value
        "sentiment_score": -9, # var with default value
        "index": data["header"]["seq"], # const
        "app_name": self.app_name, # const
        "turn": "user", # var with default value
        "last_interaction": False, # default value
        "highlighted": False, # var with default value
        "intent": self.slu_intent, # const
        "topic": self.topic_name,  # const
        "timestamp": self.timestamp, # const
        "type": "", # var
        "entity_type_detection": self.generic_entity_type_detection, # const
        "slots": data["slots"], # const
    }
    
    if self.last_user_utterance != data["utterance"]:
        self.last_user_utterance = data["utterance"]

        if len(data["utterance_sentences"]) == 0:
            if self._lastinteraction is not None and self.conversation:
                self.conversation[self._lastinteraction][
                    "lastinteraction"
                ] = False
            if self.data_keys["user"]["sentence_list"] is not None:
                self.conversation.append(
                    {
                        "idx": self.index,
                        "sentence": data["utterance"],
                        "emotion_label": "neutral",
                        "emotion_score": -9,
                        "sentiment_label": "neutral",
                        "sentiment_score": -9,
                        "index": data["header"]["seq"],
                        "app_name": self.app_name,
                        "turn": "user",
                        "last_interaction": True,
                        "highlighted": False,
                        "intent": self.slu_intent,
                        "topic": self.topic_name,
                        "timestamp": self.timestamp,
                        "type": "",
                        "entity_type_detection": self.generic_entity_type_detection,
                        "slots": data["slots"],
                    }
                )
            self.index += 1
            self._lastinteraction = 0
        else:

            self.data_keys["user"]["sentence_list"] = data["utterance_sentences"]
            if self._lastinteraction is not None and self.conversation:
                self.conversation[self._lastinteraction][
                    "lastinteraction"
                ] = False
            if self.data_keys["user"]["sentence_list"] is not None:
                for sentence_id, user in enumerate(
                    self.data_keys["user"]["sentence_list"]
                ):
                    try:
                        user_emotion_label = user["emotion_results"][
                            "emotions"
                        ]["results"]["best_match"]["label"]
                        user_emotion_score = user["emotion_results"][
                            "emotions"
                        ]["results"]["best_match"]["score"]
                    except Exception:
                        if user["emotion"] != "":
                            user_emotion_label = user["emotion"]
                            user_emotion_score = 1.0
                        elif user["auto_emotion"] != "":
                            user_emotion_label = user["auto_emotion"]
                            user_emotion_score = (
                                float(user["auto_score"])
                                if user["auto_score"] != ""
                                else 0.0
                            )
                        else:
                            user_emotion_label = (
                                self.combined_user_emotion_label
                            )
                            user_emotion_score = (
                                self.combined_user_emotion_score
                            )

                    sentiment_score = user["sentiment_results"]["sentiment"][
                        "results"
                    ]["best_match"]["score"]
                    sentiment_label = user['sentiment_results']['sentiment']['results']['best_match']['label']
                    # sentiment_label = SENTIMENT_MAPPING.get(
                    #     user["sentiment_results"]["sentiment"]["results"][
                    #         "best_match"
                    #     ]["label"],
                    #     "None",
                    # )
                    self.conversation.append(
                        {
                            "idx": self.index,
                            "sentence": user["text"],
                            "emotion_label": user_emotion_label,
                            "emotion_score": round(user_emotion_score, 2),
                            "sentiment_label": sentiment_label,
                            "sentiment_score": round(sentiment_score, 2),
                            "index": data["header"]["seq"],
                            "app_name": self.app_name,
                            "turn": "user",
                            "last_interaction": sentence_id == 0,
                            "highlighted": False,
                            "intent": self.slu_intent,
                            "topic": self.topic_name,
                            "timestamp": self.timestamp,
                            "type": user["type"],
                            "entity_type_detection": self.generic_entity_type_detection,
                            "slots": data["slots"],
                        }
                    )
                    self.index += 1
                    if sentence_id == 0:
                        self._lastinteraction = len(self.conversation) - 1
    if self.data_keys["haru"]["sentence_list"] is not None:
        for sentence_id, haru in enumerate(
            self.data_keys["haru"]["sentence_list"]
        ):
            if "|" in haru["text"]:
                continue
            try:
                haru_emotion_label = haru["emotion_results"]["emotions"][
                    "results"
                ]["best_match"]["label"]
                haru_emotion_score = haru["emotion_results"]["emotions"][
                    "results"
                ]["best_match"]["score"]
            except Exception:
                if haru["emotion"] != "":
                    haru_emotion_label = haru["emotion"]
                    haru_emotion_score = 1.0
                elif haru["auto_emotion"] != "":
                    haru_emotion_label = haru["auto_emotion"]
                    haru_emotion_score = (
                        float(haru["auto_score"])
                        if haru["auto_score"] != ""
                        else 0.0
                    )
                else:
                    haru_emotion_label = self.combined_haru_emotion_label
                    haru_emotion_score = self.combined_haru_emotion_score
            sentiment_score = haru["sentiment_results"]["sentiment"]["results"][
                "best_match"
            ]["score"]
            sentiment_label = haru['sentiment_results']['sentiment']['results']['best_match']['label']
            self.conversation.append(
                {
                    "idx": self.index,
                    "sentence": haru["text"],
                    "emotion_label": haru_emotion_label,
                    "emotion_score": round(haru_emotion_score, 2),
                    "sentiment_label": sentiment_label,
                    "sentiment_score": round(sentiment_score, 2),
                    "index": data["header"]["seq"],
                    "app_name": self.app_name,
                    "turn": "haru",
                    "last_interaction": False,
                    "highlighted": False,
                    "intent": self.slu_intent,
                    "topic": self.topic_name,
                    "timestamp": self.timestamp,
                    "type": haru["type"],
                    "entity_type_detection": self.generic_entity_type_detection,
                    "slots": data["slots"],
                }
            )
            self.index += 1
    self._data_is_ready = True
    return self.conversation