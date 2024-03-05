# #! ./env/bin/python

# from datetime import datetime
# import os
# import rospy
# import rosnode

# from time import time
# import signal
# import numpy as np
# from rich.console import Console
# from rich import print
# import json
# from threading import Lock, Thread
# from std_msgs.msg import Empty
# from strawberry_ros_msgs.msg import DialogResult

# EMOTIONS = ["anger", "disgust", "fear",
#             "joy", "neutral", "sadness", "surprise"]
# GENRE_MAPPING = {
#     "anger": "whiny",
#     "disgust": "whiny",
#     "fear": "serious",
#     "joy": "highnrg",
#     "neutral": "neutral",
#     "sadness": "sad",
#     "surprise": "highnrg"}
# SENTIMENT_MAPPING = {
#     "neg": "negative",
#     "neu": "neutral",
#     "pos": "positive",
#     }


# class Timer:
#     def __init__(self) -> None:
#         self._start_time = 0
#         self._stop_time = 0

#     def start_timer(self):
#         self._start_time = time()

#     def stop_timer(self):
#         self._stop_time = time()

#     def duration(self):
#         return self._stop_time - self._start_time


# class HaruChatCLI:
#     def __init__(self, config, sentences=None) -> None:
#         self._config = config

#         self.configure()

#         self._sentences = sentences
#         self.index = 0
#         self.last_user_utterance = ''
#         self.app_name = "smalltalk"
#         self.conversation = []
#         # Set Parameters
#         self.data_type_list = ['sentence_list', 'sentence_id', 'Sentence', 'emotion_name', 'emotion_score', 'sentiment_name', 'sentiment_score',
#                                'emotion_frequency', 'probability_emote', 'random_value', 'react', 'reaction_text', 'sentence_processed', 'sentence_processed_list']
#         self.data = {key2: {key: None for key in self.data_type_list}
#                      for key2 in ['user', 'haru']}
#         self.data['index'] = None
#         self._lastinteraction = None


#         self._use_stdin = not self._sentences or len(self._sentences) == 0

#         if not self._use_stdin:
#             self._iter = iter(self._sentences)

#         self._topics = {
#             "dialog_result": "/strawberry/dialog_result"
#         }
#         self._publisher = None

#         self._required_module = "/strawberry_ros_dialog"
#         self._required_module_name = "Strawberry ROS Dialog"
#         self._required_module_start_up_timeout = 120

#         self._send_allowed = True

#         self._records = []
#         self.mutex = Lock()

#         self._current_record = None

#         self._console = Console()

#         rospy.init_node("haru_chat_record")


#         self.init_ros()

#         signal.signal(signal.SIGINT, self._handler)

#     def configure(self):
#         if not os.path.exists(self._config["history_path"]):
#             os.makedirs(self._config["history_path"])

#     def _handler(self, signum, frame):
#         # self.report()

#         print("\nGoodbye!\n")
#         exit()
        
#     def init_ros(self):
#         rospy.Subscriber(
#             self._topics["dialog_result"],
#             DialogResult,
#             self.callback_smalltalk_dialog,
#             queue_size=1,
#         )


#     def callback_smalltalk_dialog(self, data):
#         with self.mutex:
#             self.data['index'] = data.header.seq
#             self.data['haru']['sentence_list'] = data.fulfillment_sentences

#             self.combined_haru_emotion_label = data.fulfillment_emotion.emotions.results.best_match.label
#             self.combined_haru_emotion_score = data.fulfillment_emotion.emotions.results.best_match.score
#             self.combined_user_emotion_label = data.utterance_emotion.emotions.results.best_match.label
#             self.combined_user_emotion_score = data.utterance_emotion.emotions.results.best_match.score
#             self.slu_intent = data.slu_result.intent
#             self.slu_topic_ = data.topic
#             self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#             split_parts = self.slu_topic_.split("-")
#             self.generic_entity_type_detection = False
#             if data.slots :
#                 for slots in data.slots :
#                     print("slot_name :",slots.name)
#                     if slots.name == "enable_intent_classification" :
#                         self.generic_entity_type_detection = True
#                     else :
#                         self.generic_entity_type_detection = False


            
#             if len(split_parts) >= 2:
#                 self.topic_name = " ".join(split_parts[1:])
#                 self.slu_topic = split_parts[0]
#             else :
#                 self.topic_name = self.slu_topic_
#                 self.slu_topic = self.slu_topic_


            
#             if self.last_user_utterance != data.utterance :
#                 self.last_user_utterance = data.utterance

#                 # self.combined_haru_sentiment_label = data.sentiment_results.fulfillment_sentiment.sentiment.results.best_match.label
#                 # self.combined_haru_sentiment_score = data.sentiment_results.fulfillment_sentiment.sentiment.results.best_match.score
#                 # self.combined_user_sentiment_label = data.sentiment_results.utterance_sentiment.sentiment.results.best_match.label
#                 # self.combined_user_sentiment_score = data.sentiment_results.utterance_sentiment.sentiment.results.best_match.score
#                 if len(data.utterance_sentences) == 0:
#                     if self._lastinteraction is not None and self.conversation != []:
#                         self.conversation[self._lastinteraction]['lastinteraction'] = False
#                     if self.data['user']['sentence_list'] is not None:
#                         self.conversation.append({
#                             'idx': self.index,
#                             'Sentence': data.utterance,
#                             'emotion_label': "neutral",
#                             'emotion_score': -9,
#                             'sentiment_label': "neutral",
#                             'sentiment_score': -9,
#                             'index': data.header.seq,
#                             'app_name': self.app_name,
#                             'Turn': 'user',
#                             'LastInteraction': True,
#                             'Highlighted': False,
#                             'Intent': self.slu_intent,
#                             'intent_category':self.slu_topic,
#                             'timestamp' : self.timestamp,
#                             'topic_name' : self.topic_name,
#                             'type' : '',
#                             'entity_type_detection':self.generic_entity_type_detection,
#                         })
#                         self.index += 1
#                         self._lastinteraction = 0
#                 else:
#                     self.data['user']['sentence_list'] = data.utterance_sentences
#                     if self._lastinteraction is not None and self.conversation != []:
#                         self.conversation[self._lastinteraction]['lastinteraction'] = False
#                     if self.data['user']['sentence_list'] is not None:
#                         for sentence_id, user in enumerate(self.data['user']['sentence_list']):
#                             try:
                                    
#                                 rospy.loginfo(
#                                     "Using original emotion instead of rich response msg")
#                                 user_emotion_label = user.emotion_results.emotions.results.best_match.label
#                                 user_emotion_score = user.emotion_results.emotions.results.best_match.score
#                             except Exception:
#                                 if user.emotion != "":
#                                     user_emotion_label = user.emotion
#                                     user_emotion_score = 1.0
#                                 elif user.auto_emotion != "":
#                                     user_emotion_label = user.auto_emotion
#                                     user_emotion_score = float(
#                                         user.auto_score) if user.auto_score != "" else 0.0
#                                 else:
#                                     user_emotion_label = self.combined_user_emotion_label
#                                     user_emotion_score = self.combined_user_emotion_score

#                             sentiment_score = user.sentiment_results.sentiment.results.best_match.score
#                             sentiment_label = SENTIMENT_MAPPING.get(
#                                 user.sentiment_results.sentiment.results.best_match.label, "None")
#                             self.conversation.append({
#                                 'idx': self.index,
#                                 'Sentence': user.text,
#                                 'emotion_label': user_emotion_label,
#                                 'emotion_score': np.round(user_emotion_score, 2),
#                                 'sentiment_label': sentiment_label,
#                                 'sentiment_score': np.round(sentiment_score, 2),
#                                 'index': data.header.seq,
#                                 'app_name': self.app_name,
#                                 'Turn': 'user',
#                                 'lastinteraction': sentence_id == 0,
#                                 'Highlighted': False,
#                                 'Intent': self.slu_intent,
#                                 'intent_category':self.slu_topic,
#                                 'timestamp' : self.timestamp,
#                                 'topic_name' : self.topic_name,
#                                 'type' : user.type,
#                                 'entity_type_detection':self.generic_entity_type_detection,
#                             })
#                             self.index += 1
#                             if sentence_id == 0:
#                                 self._lastinteraction = len(self.conversation)-1
#             if self.data['haru']['sentence_list'] is not None:
#                 for sentence_id, haru in enumerate(self.data['haru']['sentence_list']):
#                     if '|' in haru.text:
#                         continue
#                     try:    
#                         rospy.loginfo(
#                             "Using original emotion instead of rich response msg")
#                         haru_emotion_label = haru.emotion_results.emotions.results.best_match.label
#                         haru_emotion_score = haru.emotion_results.emotions.results.best_match.score
#                     except Exception:
#                         if haru.emotion != "":
#                             haru_emotion_label = haru.emotion
#                             haru_emotion_score = 1.0
#                         elif haru.auto_emotion != "":
#                             haru_emotion_label = haru.auto_emotion
#                             haru_emotion_score = float(
#                                 haru.auto_score) if haru.auto_score != "" else 0.0
#                         else:
#                             haru_emotion_label = self.combined_haru_emotion_label
#                             haru_emotion_score = self.combined_haru_emotion_score
#                     sentiment_score = haru.sentiment_results.sentiment.results.best_match.score
#                     sentiment_label = SENTIMENT_MAPPING.get(
#                         haru.sentiment_results.sentiment.results.best_match.label, "None")
#                     self.conversation.append({
#                         'idx': self.index,
#                         'Sentence': haru.text,
#                         'emotion_label': haru_emotion_label,
#                         'emotion_score': np.round(haru_emotion_score, 2),
#                         'sentiment_label': sentiment_label,
#                         'sentiment_score': np.round(sentiment_score, 2),
#                         'index': data.header.seq,
#                         'app_name': self.app_name,
#                         'Turn': 'haru',
#                         'lastinteraction': False,
#                         'Highlighted': False,
#                         'Intent': self.slu_intent,
#                         'intent_category':self.slu_topic,
#                         'timestamp' : self.timestamp,
#                         'topic_name' : self.topic_name,
#                         'type' : haru.type,
#                         'entity_type_detection':self.generic_entity_type_detection,
#                     })
#                     self.index += 1
#             self._data_is_ready = True
            
#         # Specify the file path
#         file_path = 'conversations.json'

#         # Write the list to a JSON file
#         with open(file_path, 'a') as f:
#             json.dump(self.conversation, f)
            
            
#     def run(self, language_code=None):
#         language_code = (
#             language_code if language_code else self._config.get("language_code", "en")
#         )

#         first_pass = True

#         while True:
#             rospy.spin()
#             pass
#             # if not first_pass:
#             #     print("\n")

#             # while not self._send_allowed:
#             #     try:
#             #         sleep(0.2)
#             #     except KeyboardInterrupt as e:
#             #         exit(1)
#             # if self._use_stdin:
#             #     self._console.print("> ", style="bold rgb(255,255,100)", end="")
#             #     # text = input()
#             # else:
#             #     try:
#             #         text = next(self._iter)
#             #         sleep(0.2)
#             #         self._console.print("> ", style="bold rgb(255,100,100)", end="")
#             #         self._console.print(f"{text}", style="bold rgb(255,100,100)")
#             #     except Exception as e:
#             #         break
                
        


# if __name__ == "__main__":
    
#     console = Console()

#     try:
#         node_name = "/haru_smalltalk_record"
#         config = {
#             "history_path": rospy.get_param(
#                 f"{node_name}/history_path",
#                 f"{os.environ.get('HOME')}/.ros/haru-project/haru-smalltalk-cli",
#             ),
#             "language_code": "en",
#         }
#         cli = HaruChatCLI(config)
#         cli.run()

#     except Exception as e:
#         console.print(e)
#     finally:
#         console.print("\nShutdown OK", style=f"bold rgb(100,255,100)")
#         console.print("GOODBYE", style=f"bold rgb(100,100,255)")