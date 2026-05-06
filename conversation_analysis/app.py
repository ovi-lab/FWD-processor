import os
from flask import Flask
from flask_socketio import SocketIO
import json
from datetime import datetime
from threading import Event, Lock

import yaml
from process_uploaded_transcript import process_uploaded_transcript
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

def intercepting_file(file_name):
    ext = os.path.splitext(file_name)[-1].lower()

    if ext == ".json":
        return TRANSCRIPT_FILE_PATH, True, False

    if ext == ".txt":
        return UPLOADED_FILE_PATH, False, True

    return UPLOADED_FILE_PATH, False, False

@socketio.event
def upload_file(file_data, file_name):
    """
    Handles file uploads from clients, determining the file path and processing the uploaded transcript.
    This function kills any active ROS feed before processing to avoid conflicts with incoming data.
    """
    kill_ros_feed()
    if file_data: 
        save_path, is_json, is_txt = intercepting_file(file_name)
        with open(save_path, 'wb') as f:
            f.write(file_data)
        process_uploaded_transcript(save_path, is_json=is_json, is_txt=is_txt)

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
def transcript_request(retried=0):
    """
    Responds to client requests for the transcript. It retries up to 2 times if the file is not found,
    which could be the case if the file is still being processed or needs to be remade.
    """
    try:
        with open(TRANSCRIPT_FILE_PATH, 'r', encoding='utf8') as file:
            payload = json.load(file)
        socketio.emit('transcript_response', payload)
    except FileNotFoundError:
        if retried < 2:
            process_uploaded_transcript(UPLOADED_FILE_PATH, False)
            transcript_request(retried + 1)
        else:
            socketio.emit('unable-to-open', TRANSCRIPT_FILE_PATH)


@socketio.event
def diagram_request(retried=0):
    """
    Responds to client requests for data needed to generate the diagrams. It retries up to 2 times if the file is not found,
    which could be the case if the file is still being processed or needs to be remade.
    """
    try:
        with open(DIAGRAM_FILE_PATH, 'r', encoding='utf8') as file:
            data = json.load(file)
        socketio.emit('diagram_response', data)
    except FileNotFoundError:
        if retried < 2:
            process_uploaded_transcript(UPLOADED_FILE_PATH, False)
            diagram_request(retried + 1)
        else:
            socketio.emit('unable-to-open', DIAGRAM_FILE_PATH)



if __name__ == "__main__":
    # Start the WebSocket server
    socketio.run(app, debug=False, port=6400, use_reloader=False, log_output=False)
