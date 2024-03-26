from  process_utils.parse_yml_output import parse_yml_output
from process_utils.clean_JSON import process_conversation
from process_utils.diagram_prep.diagram_process import generate_diagram_from_file

def process_uploaded_transcript(ros_transcript):
    """
    Takes a full list of ROS nodes and converts it into 
    JSON for the Visualization App
    """
    data = parse_yml_output(ros_transcript)
    if not data:
        print('error')
        return
    chat_json_location = f'data/transcript_data/transcript-log.json'
    diagram_json_location = f'data/diagram_data/diagram-log.json'
    processed_transcript = process_conversation(data, chat_json_location)
    generate_diagram_from_file(processed_transcript, diagram_json_location)
    return

    

if __name__ == '__main__':
    
    process_uploaded_transcript('data/raw_uploads/dialog_result.yml', 'log-03-12.json')