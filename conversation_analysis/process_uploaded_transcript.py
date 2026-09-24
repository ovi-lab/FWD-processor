import os
import re
import json
import threading
from process_utils.parse_yml_output import parse_yml_output
from process_utils.clean_JSON import process_conversation, process_from_json
from process_utils.diagram_prep.diagram_process import generate_diagram_from_file
from raw_transcript_converter import RawTranscriptConverter

# Paths are anchored to the repo root so this works from app.py and from any working directory.
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
diagram_json_location = os.path.join(DATA_DIR, 'diagram_data', 'diagram-log.json')
chat_json_location = os.path.join(DATA_DIR, 'transcript_data', 'transcript-log.json')

RAW_UPLOADS_DIR = os.path.join(DATA_DIR, 'raw_uploads')
NAMED_OUTPUTS_DIR = os.path.join(DATA_DIR, 'named_outputs')
PROCESSED_PREFIX = 'processed_'

# Lazy singleton — models are only loaded when a transcript actually needs processing.
# Diagram-only paths never touch this, so smaller machines can run without ML deps.
_converter = None
# The server's startup warm-up and an early upload can both ask for the converter at once;
# without this lock each would load its own copy of the models.
_converter_lock = threading.Lock()

def _get_converter():
    global _converter
    with _converter_lock:
        if _converter is None:
            _converter = RawTranscriptConverter()
    return _converter


def _extract_metadata_from_filename(filename):
    """Extract participant ID and child name from filenames like 'P1_Bob.json' or '16_Raiken.json'."""
    name, ext = os.path.splitext(filename)
    if ext.lower() not in ('.json', '.txt', ''):
        name = filename  # 'P19.1-Alex' has no extension; its '.1-Alex' isn't one
    # Also accepts split sessions like 'P19.1-Alex' (ID P19.1).
    match = re.match(r'^P?(\d+(?:\.\d+)?)[_-](.+)$', name, re.IGNORECASE)
    if match:
        return f'P{match.group(1)}', match.group(2)
    # Name first, then the number: 'Lucy5.1' -> P5.1 / Lucy
    match = re.match(r'^([A-Za-z]+)[_-]?(\d+(?:\.\d+)?)$', name)
    if match:
        return f'P{match.group(2)}', match.group(1)
    return None, None


def _is_haru_json_format(json_data):
    """Returns True if the JSON looks like the Haru time/haru/user format."""
    return (
        isinstance(json_data, list)
        and len(json_data) > 0
        and isinstance(json_data[0], dict)
        and "haru" in json_data[0]
    )


def process_haru_json(json_data, filename):
    """
    Runs the full pipeline on a raw Haru JSON transcript: emotion, sentiment and topics,
    then writes the transcript log, the diagram data and data/named_outputs/<ID>_<Name>.json.
    Used by both process_new_uploads() and the browser upload in app.py.
    Returns (processed_json, output_filename).
    """
    participant_id, child_name = _extract_metadata_from_filename(filename)
    if not child_name:
        _, child_name = _get_converter().extract_metadata_from_json(json_data)
    if not participant_id:
        participant_id = 'P0'
    if not child_name:
        child_name = filename  # full name, so 'Lucy5.1' and 'Lucy5.2' can't overwrite each other

    print(f'Processing: {filename} (ID: {participant_id}, Name: {child_name})')
    data = _get_converter().convert_json_transcript(json_data)

    processed_json = process_conversation(data, chat_json_location)
    generate_diagram_from_file(processed_json, diagram_json_location)

    os.makedirs(NAMED_OUTPUTS_DIR, exist_ok=True)
    output_filename = f'{participant_id}_{child_name}.json'
    with open(os.path.join(NAMED_OUTPUTS_DIR, output_filename), 'w', encoding='utf-8') as f:
        json.dump(processed_json, f, indent=4, ensure_ascii=False)

    print(f'Saved: {output_filename}')
    return processed_json, output_filename


def process_uploaded_transcript(upload_path, is_json=False, is_txt=False):
    """
    Converts uploaded transcript files into JSON for the Visualization App.
    Supports YAML, JSON, and TXT.
    """

    if is_txt:
        data = _get_converter().convert_file(upload_path)
        processed_json = process_conversation(data, chat_json_location)
        generate_diagram_from_file(processed_json, diagram_json_location)

    elif is_json:
        with open(upload_path, 'r', encoding='utf8') as json_file:
            transcript_data = json.load(json_file)

        if _is_haru_json_format(transcript_data):
            data = _get_converter().convert_json_transcript(transcript_data)
            processed_json = process_conversation(data, chat_json_location)
        else:
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


def process_new_uploads():
    """
    Scans raw_uploads for unprocessed TXT files, extracts participant ID
    and child name from the top of the file (e.g. {P1} and {John}),
    processes each one, saves named output to named_outputs/P1_John.json,
    and renames the source file with a 'processed_' prefix so it won't
    be picked up again.
    """

    os.makedirs(NAMED_OUTPUTS_DIR, exist_ok=True)

    all_files = os.listdir(RAW_UPLOADS_DIR)
    raw_txt = [f for f in all_files if f.endswith('.txt') and not f.startswith(PROCESSED_PREFIX)]
    raw_json = [f for f in all_files if f.endswith('.json') and not f.startswith(PROCESSED_PREFIX)]
    raw_files = [(f, 'txt') for f in raw_txt] + [(f, 'json') for f in raw_json]

    if not raw_files:
        print('No new files to process.')
        return

    for filename, file_type in raw_files:
        upload_path = os.path.join(RAW_UPLOADS_DIR, filename)

        try:
            if file_type == 'txt':
                print(f'Reading metadata from: {filename}')
                participant_id, child_name = _get_converter().extract_metadata(upload_path)

                if not participant_id or not child_name:
                    print(f'Skipping {filename} — could not find {{P1}} and {{Name}} at top of file')
                    continue

                print(f'Processing: {filename} (ID: {participant_id}, Name: {child_name})')
                data = _get_converter().convert_file(upload_path)

                processed_json = process_conversation(data, chat_json_location)
                generate_diagram_from_file(processed_json, diagram_json_location)

                output_filename = f'{participant_id}_{child_name}.json'
                with open(os.path.join(NAMED_OUTPUTS_DIR, output_filename), 'w', encoding='utf-8') as f:
                    json.dump(processed_json, f, indent=4, ensure_ascii=False)

                print(f'Saved: {output_filename}')

            else:
                print(f'Reading JSON transcript: {filename}')
                with open(upload_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                process_haru_json(json_data, filename)

            processed_path = os.path.join(RAW_UPLOADS_DIR, PROCESSED_PREFIX + filename)
            os.rename(upload_path, processed_path)
            print(f'Marked as processed: {PROCESSED_PREFIX + filename}')

        except Exception as e:
            print(f'Error processing {filename}: {e}')


if __name__ == '__main__':
    process_new_uploads()
