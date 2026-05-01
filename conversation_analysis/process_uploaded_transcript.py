from  process_utils.parse_yml_output import parse_yml_output
from process_utils.clean_JSON import process_conversation, process_from_json
from process_utils.diagram_prep.diagram_process import generate_diagram_from_file
from raw_transcript_converter import RawTranscriptConverter
import json

diagram_json_location = f'data/diagram_data/diagram-log.json'
chat_json_location = f'data/transcript_data/transcript-log.json'

def process_uploaded_transcript(upload_path, is_json=False, is_txt=False):
    """
    Converts uploaded transcript files into JSON for the Visualization App.
    Supports YAML, JSON, and TXT.
    """

    if is_txt:
        converter = RawTranscriptConverter(max_topics=10)
        data = converter.convert_file(upload_path)
        processed_json = process_conversation(data, chat_json_location)
        generate_diagram_from_file(processed_json, diagram_json_location)

    elif is_json:
        with open(upload_path, 'r', encoding='utf8') as json_file:
            transcript_data = json.load(json_file)

        processed_json = process_from_json(transcript_data, chat_json_location)
        generate_diagram_from_file(processed_json, diagram_json_location)

    else:
        data = parse_yml_output(upload_path)

        if not data:
            print('error')
            return

        processed_json = process_conversation(data, chat_json_location)
        generate_diagram_from_file(processed_json, diagram_json_location)

    return processed_json

    

if __name__ == '__main__':
    
     process_uploaded_transcript(
        'data/raw_uploads/sample_transcript.txt',
        is_txt=True
    )