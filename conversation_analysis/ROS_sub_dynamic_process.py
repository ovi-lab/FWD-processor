#! ./env/bin/python

from datetime import datetime
import os
import roslibpy
# import rosnode

from time import time
import signal
from rich.console import Console
from rich import print
import json
from threading import Lock
from process_utils.parse_subscriber_data import parse_subscriber_data

EMOTIONS = ["anger", "disgust", "fear",
            "joy", "neutral", "sadness", "surprise"]
GENRE_MAPPING = {
    "anger": "whiny",
    "disgust": "whiny",
    "fear": "serious",
    "joy": "highnrg",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": "highnrg"}
# SENTIMENT_MAPPING = {
#     "neg": "negative",
#     "neu": "neutral",
#     "pos": "positive",
#     }


class Timer:
    def __init__(self) -> None:
        self._start_time = 0
        self._stop_time = 0

    def start_timer(self):
        self._start_time = time()

    def stop_timer(self):
        self._stop_time = time()

    def duration(self):
        return self._stop_time - self._start_time


class RobotChatCLI:
    """
    Robot subscriber object created originally by Lithin with minor modifications.
    Gets the raw ROS outputs and adds them to a json file
    """
    def __init__(self, config, sentences=None) -> None:
        self._config = config

        # self.configure()

        self.client = roslibpy.Ros(host='10.80.58.230', port=9090)
        self.client.run()

        self._sentences = sentences
        self.index = 0
        self.last_user_utterance = ''
        self.app_name = "smalltalk"
        self.conversation = []
        # Set Parameters
        self.data_type_list = ['sentence_list', 'sentence_id', 'sentence', 'emotion_name', 'emotion_score', 'sentiment_name', 'sentiment_score',
                               'emotion_frequency', 'probability_emote', 'random_value', 'react', 'reaction_text', 'sentence_processed', 'sentence_processed_list']
        self.data_keys = {key2: {key: None for key in self.data_type_list}
                     for key2 in ['user', 'robot']}
        self.data_keys['index'] = None
        self._lastinteraction = None
        self.prev_topic = ''

        self._use_stdin = not self._sentences or len(self._sentences) == 0

        if not self._use_stdin:
            self._iter = iter(self._sentences)

        self._topics = {
            "dialog_result": "/strawberry/dialog_result"
        }
        self._publisher = None

        self._required_module = "/strawberry_ros_dialog"
        self._required_module_name = "Strawberry ROS Dialog"
        self._required_module_start_up_timeout = 120

        self._send_allowed = True

        self._records = []
        self.mutex = Lock()

        self._current_record = None

        self._console = Console()

        # rospy.init_node("haru_chat_record")


        self.init_ros()

        signal.signal(signal.SIGINT, self._handler)

    def configure(self):
        if not os.path.exists(self._config["history_path"]):
            os.makedirs(self._config["history_path"])

    def _handler(self, signum, frame):
        # self.report()

        print("\nGoodbye!\n")
        exit()
        
    def init_ros(self):
        listener = roslibpy.Topic(self.client, '/strawberry/dialog_result', 'strawberry_ros_msgs/DialogResult')
        listener.subscribe(self.callback_smalltalk_dialog)


    def callback_smalltalk_dialog(self, data):
        with self.mutex:
            self.conversation = parse_subscriber_data(self, data)
            
        # Specify the file path
        BASE_DIR = os.path.dirname(os.path.dirname(__file__))
        DATA_DIR = os.path.join(BASE_DIR, 'data')
        TRANSCRIPT_FILE_PATH = os.path.join(DATA_DIR, 'transcript_data', 'transcript-log.json')        
        RAW_RESULT_PATH = os.path.join(DATA_DIR, 'raw_YAML', 'dialog_result.json')
        
        # Write the transcript to a JSON file
        with open(TRANSCRIPT_FILE_PATH, 'w') as transcript_file:
            json.dump(self.conversation, transcript_file, indent=4)

        with open(RAW_RESULT_PATH, 'w') as raw_file:
            json.dump(data, raw_file, indent=4)
            
    def run(self, language_code=None):
        language_code = (
            language_code if language_code else self._config.get("language_code", "en")
        )

        first_pass = True

        while True:
            # rospy.spin()
            pass
                
        
def connect_to_ROS(): 
    console = Console()

    ### run the ROS server at ~/haru-repos/new_topic_ws with the alias
    #       dlg 
    ### run the ROS CLI  ~/haru-repos/new_topic_ws with the alias
    #       cli

    ### Before running this, make sure you source the workspace to the setup:
    #       source /home/lithin/haru-repos/new_topics_ws/devel/setup.bash
    ### then set the listener:
    #       rostopic echo /strawberry/dialog_result
    ### now you can run this script in the sourced location

    try:
        node_name = "/haru_smalltalk_record"
        # config = {
        #     "history_path": rospy.get_param(
        #         f"{node_name}/history_path",
        #         f"{os.environ.get('HOME')}/.ros/haru-project/haru-smalltalk-cli",
        #     ),
        #     "language_code": "en",
        # }
        config={}
        cli = RobotChatCLI(config)
        cli.run()

    except Exception as e:
        console.print(e)
    finally:
        console.print("\nShutdown OK", style=f"bold rgb(100,255,100)")
        console.print("GOODBYE", style=f"bold rgb(100,100,255)")

if __name__ == "__main__":
    
    connect_to_ROS()
