from  static_processing.parse_ros_output import parse_ros_output
from static_processing.clean_JSON import process_conversation
from diagram_prep.diagram_process import diagram_process

def process_static_transcript(ros_transcript, json_output_name):
    """
    Takes a full list of ROS nodes and converts it into 
    JSON for the Visualization App
    """
    data = parse_ros_output(ros_transcript)
    if not data:
        print('error')
        return
    chat_json_location = f'data/prepped_for_chats/chat-{json_output_name}'
    diagram_json_location = f'data/prepped_for_diagrams/diagram-{json_output_name}'
    process_conversation(data, chat_json_location)
    diagram_process(chat_json_location, diagram_json_location)

    

if __name__ == '__main__':
    
    process_static_transcript('data/raw_uploads/dialog_result.yml', 'log-03-12.json')