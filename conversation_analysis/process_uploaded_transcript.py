from  process_utils.parse_yml_output import parse_yml_output
from process_utils.clean_JSON import process_conversation
from process_utils.diagram_prep.diagram_process import generate_diagram_from_file
import json

diagram_json_location = f'data/diagram_data/diagram-log.json'

def process_uploaded_transcript(upload_path, is_json):
    """
    Takes a full list of ROS nodes and converts it into 
    JSON for the Visualization App
    """
    if not is_json:
        data = parse_yml_output(upload_path)
        if not data:
            print('error')
            return
        chat_json_location = f'data/transcript_data/transcript-log.json'
        processed_transcript = process_conversation(data, chat_json_location)
        generate_diagram_from_file(processed_transcript, diagram_json_location)
    else:
        with open(upload_path, 'r', encoding='utf8') as json_file:
            transcript_data = json.load(json_file)
            generate_diagram_from_file(transcript_data, diagram_json_location)
    return

    

if __name__ == '__main__':
    
    process_uploaded_transcript('data/raw_uploads/dialog_result.yml', 'log-03-12.json')