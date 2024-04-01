from  process_utils.parse_yml_output import parse_yml_output
from process_utils.clean_JSON import process_conversation, process_from_json
from process_utils.diagram_prep.diagram_process import generate_diagram_from_file
import json

diagram_json_location = f'data/diagram_data/diagram-log.json'
chat_json_location = f'data/transcript_data/transcript-log.json'

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
        processed_transcript = process_conversation(data, chat_json_location)
        generate_diagram_from_file(processed_transcript, diagram_json_location)
    else:
        with open(upload_path, 'r', encoding='utf8') as json_file:
            transcript_data = json.load(json_file)
            processed_json = process_from_json(transcript_data, chat_json_location)
            generate_diagram_from_file(processed_json, diagram_json_location)
    return processed_json

    

if __name__ == '__main__':
    
    process_uploaded_transcript('data/raw_uploads/dialog_result.yml', 'log-03-12.json')