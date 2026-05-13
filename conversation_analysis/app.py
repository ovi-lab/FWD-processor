import os
from flask import Flask
from flask_socketio import SocketIO, emit
import json
from datetime import datetime
from threading import Event, Lock

import yaml
from process_utils.clean_JSON import process_from_json
from process_utils.diagram_prep.diagram_process import generate_diagram_from_file


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

# Setting up the paths needed for the application
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
app.config['UPLOAD_FOLDER'] = os.path.join(DATA_DIR, 'tmp')
TRANSCRIPT_FILE_PATH = os.path.join(DATA_DIR, 'transcript_data', 'transcript-log.json')
DIAGRAM_FILE_PATH = os.path.join(DATA_DIR, 'diagram_data', 'diagram-log.json')
UPLOADED_FILE_PATH = os.path.join(app.config['UPLOAD_FOLDER'], 'raw_log.yml')

# Thread controls for ROS feed
thread_event = Event()
thread_lock = Lock()
thread = None

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


@socketio.event
def init_ros_feed():
    """
    Initializes the background task for receiving ROS feed. It checks if there's already a thread running and starts
    one if not.
    """
    global thread
    with thread_lock:
        if thread is None:
            thread_event.set()
            thread = socketio.start_background_task(receive_subscriber_feed, thread_event)
    print("Realtime feed to visualizer initialized")

@socketio.event
def kill_ros_feed():
    """
    Stops the ROS feed by clearing the event and joining the thread if it exists, closing without leaving hanging processes.
    """
    global thread
    thread_event.clear()
    with thread_lock:
        if thread is not None:
            thread.join()
            thread = None
    print('Realtime feed to visualizer ended')

def receive_subscriber_feed(event):
    """
    Continuously checks for new updates in the transcript file. If a new update is found, it emits the updated data
    to connected clients. It also generates and emits corresponding diagram data. The function runs as a background
    thread that updates clients with new data at set intervals.
    """
    transcript_last_idx = -1
    global thread
    try:
        while event.is_set():
            with open(TRANSCRIPT_FILE_PATH, 'r', encoding='utf8') as transcript_file:
                transcript_payload = json.load(transcript_file)
                if (transcript_payload[-1]['idx'] != transcript_last_idx):
                    print(transcript_payload[-1]['idx'])
                    print('updating...')
                    transcript_last_idx = transcript_payload[-1]['idx']
                    socketio.emit('transcript_response', transcript_payload)
                    diagram_payload = generate_diagram_from_file(transcript_payload, DIAGRAM_FILE_PATH)
                    socketio.emit('diagram_response', diagram_payload)
                socketio.sleep(1) # This sleep function adjusts the polling rate for dialogue updates
            print('waiting')
    except Exception as e:
        print(f'Exception on emission: {e}')
        socketio.emit('fatal-emission', e)
    finally:
        thread_event.clear()
        thread = None

def _is_processed_transcript(json_data):
    """Returns True if the JSON is a processed transcript (has the fields the frontend expects)."""
    required = {'idx', 'sentence', 'turn', 'slots', 'emotion_label', 'sentiment_label'}
    return (
        isinstance(json_data, list)
        and len(json_data) > 0
        and isinstance(json_data[0], dict)
        and required.issubset(json_data[0].keys())
    )


@socketio.event
def upload_file(file_data, file_name):
    """
    Accepts pre-processed transcript JSON files only.
    Raw transcript processing is handled separately by process_uploaded_transcript.py.
    """
    kill_ros_feed()
    if not file_data:
        return

    try:
        ext = os.path.splitext(file_name)[-1].lower()
        if ext != '.json':
            emit('upload-error', 'Only pre-processed .json files are supported here. Use process_uploaded_transcript.py for raw transcripts.')
            return

        # Decode bytes to string (utf-8-sig strips BOM if present)
        if isinstance(file_data, (bytes, bytearray)):
            content = file_data.decode('utf-8-sig')
        else:
            content = file_data.lstrip('﻿')

        transcript_data = json.loads(content)

        if not _is_processed_transcript(transcript_data):
            emit('upload-error', 'This file has not been processed yet. Run process_uploaded_transcript.py first, then upload the file from data/named_outputs/.')
            return

        with open(TRANSCRIPT_FILE_PATH, 'w', encoding='utf8') as f:
            json.dump(transcript_data, f, indent=4)

        processed_json = process_from_json(transcript_data, TRANSCRIPT_FILE_PATH)
        generate_diagram_from_file(processed_json, DIAGRAM_FILE_PATH)
        emit('upload-success')
    except Exception as e:
        print(f'Upload error: {e}')
        emit('upload-error', str(e))

@socketio.event
def json_download_request():
    """
    Allows clients to request the download of the transcript log in JSON format 
    (processed for the application to use). 
    It provides the file with a timestamp to differentiate it from other downloads.
    """
    try:
        with open(TRANSCRIPT_FILE_PATH, 'r', encoding='utf8') as file:
            payload = json.load(file)
        time_value = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        file_name = f"transcript-log_{time_value}.json"
        socketio.emit('json_download_response', [payload, file_name])
        print(f"{file_name} sent for download")
    except FileNotFoundError:
        print(FileNotFoundError)

@socketio.event
def yaml_download_request():
    """
    Allows clients to request the download of the transcript log in YAML format 
    (what ROS outputs). 
    It provides the file with a timestamp to differentiate it from other downloads.
    """
    try:
        with open(UPLOADED_FILE_PATH, 'r', encoding='utf8') as file:
            yaml_data =  yaml.safe_load_all(file)
            payload = '\n---\n'.join([yaml.dump(data) for data in yaml_data])
        time_value = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        file_name = f"transcript-log_{time_value}.yaml"
        socketio.emit('yaml_download_response', [payload, file_name])
        print(f"{file_name} sent for download")
    except FileNotFoundError:
        print(FileNotFoundError)

@socketio.event
def transcript_request():
    """
    Responds to client requests for the transcript JSON.
    The file is produced by process_uploaded_transcript.py on the processing machine.
    """
    try:
        with open(TRANSCRIPT_FILE_PATH, 'r', encoding='utf8') as file:
            payload = json.load(file)
        emit('transcript_response', payload)
    except FileNotFoundError:
        emit('unable-to-open', TRANSCRIPT_FILE_PATH)


@socketio.event
def diagram_request():
    """
    Responds to client requests for diagram data.
    The file is produced by process_uploaded_transcript.py on the processing machine.
    """
    try:
        with open(DIAGRAM_FILE_PATH, 'r', encoding='utf8') as file:
            data = json.load(file)
        emit('diagram_response', data)
    except FileNotFoundError:
        emit('unable-to-open', DIAGRAM_FILE_PATH)



if __name__ == "__main__":
    # Start the WebSocket server
    socketio.run(app, debug=False, port=6400, use_reloader=False, log_output=False)
