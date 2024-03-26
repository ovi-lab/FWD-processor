import os
from flask import Flask
from flask_socketio import SocketIO
import json
from threading import Event, Lock
from process_uploaded_transcript import process_uploaded_transcript
from process_utils.diagram_prep.diagram_process import generate_diagram_from_file


thread_event = Event()
thread_lock = Lock()
thread = None

data_emission_mode = 'dynamic'

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

# Setting up the paths needed for the application
dirname = os.path.dirname
DATA_DIRECTORY = os.path.join(dirname(dirname(__file__)), 'data')
app.config['UPLOAD_FOLDER'] = os.path.join(DATA_DIRECTORY, 'tmp')
TRANSCRIPT_FILE_PATH = os.path.join(DATA_DIRECTORY, 'transcript_data/transcript-log.json')
DIAGRAM_FILE_PATH = os.path.join(DATA_DIRECTORY, 'diagram_data/diagram-log.json')
UPLOADED_FILE_PATH = os.path.join(app.config['UPLOAD_FOLDER'], 'raw_log.yml')

@socketio.event
def initRosFeed():
    global thread
    with thread_lock:
        if thread is None:
            thread_event.set()
            thread = socketio.start_background_task(receiveSubscriberFeed, thread_event)

    pass

@socketio.event
def killRosFeed():
    global thread
    thread_event.clear()
    with thread_lock:
        if thread is not None:
            thread.join()
            thread = None
            print('kachow')

def receiveSubscriberFeed(event):
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
                    socketio.emit('transcript-response', transcript_payload)
                    diagram_payload = generate_diagram_from_file(transcript_payload, DIAGRAM_FILE_PATH)
                    socketio.emit('diagram-response-all', diagram_payload)
                socketio.sleep(2) 
            print('waiting')
    except Exception as e:
        print(f'Exception on emission: {e}')
        socketio.emit('fatal-emission', e)
    finally:
        event.clear()
        thread = None

def intercepting_json(file_name):
    ext = os.path.splitext(file_name)[-1].lower()
    if ext == ".json":
        return TRANSCRIPT_FILE_PATH, True
    else:
        return UPLOADED_FILE_PATH, False

@socketio.event
def upload_file(file_data, file_name):
    data_emission_mode = 'static'
    if file_data:
        print(file_name)
        save_path, isJson = intercepting_json(file_name)
        with open(save_path, 'wb') as f:
            f.write(file_data)
        process_uploaded_transcript(save_path, isJson)

@socketio.on('transcript-request')
def getLinearDialogue(retried=0):
    try:
        with open(TRANSCRIPT_FILE_PATH, 'r', encoding='utf8') as file:
            payload = json.load(file)
        socketio.emit('transcript-response', payload)
        retried = 0
    except FileNotFoundError:
        if retried==2:
            socketio.emit('unable-to-open', TRANSCRIPT_FILE_PATH)
            return
        process_uploaded_transcript(UPLOADED_FILE_PATH, False)
        getLinearDialogue(retried=(retried+1))

@socketio.on('diagram-request')
def getDiagramData(query='', retried=0):
    try:
        with open(DIAGRAM_FILE_PATH, 'r', encoding='utf8') as file:
            data = json.load(file)
        socketio.emit('diagram-response-all', data)
        retried = 0
    except FileNotFoundError:
        if retried == 2:
            socketio.emit('unable-to-open', DIAGRAM_FILE_PATH)
            return
        process_uploaded_transcript(UPLOADED_FILE_PATH, False)
        getDiagramData(query, retried=(retried+1))


@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')



if __name__ == "__main__":
    # Start the WebSocket server
    socketio.run(app, debug=False, port=6400)
