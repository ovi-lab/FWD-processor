import os
from flask import Flask
from flask_socketio import SocketIO
import json
import time
from process_uploaded_transcript import process_uploaded_transcript
from process_utils.diagram_prep.diagram_process import generate_diagram_from_file

CHAT_PATH = "data/transcript_data/chat-log-03-12.json"
DIAGRAM_PATH = "data/diagram_data/diagram-log-03-12.json"

data_emission_mode = 'dynamic'

app = Flask(__name__)
app.config['DATA_FOLDER'] = '../data/'
socketio = SocketIO(app, cors_allowed_origins='*')

uploaded_file_path = os.path.join(app.config['DATA_FOLDER'], 'tmp/upload.yml')
processed_file_suffix = 'log.json'

@socketio.on('initSubscriber')
def init_subscriber():
    data_emission_mode = 'dynamic'
    pass

@socketio.event
def writeUpload(file_data):
    data_emission_mode = 'static'
    if file_data:

        # Write the file data to the file
        with open(uploaded_file_path, 'wb') as f:
            f.write(file_data)
        process_uploaded_transcript(uploaded_file_path, processed_file_suffix)

@socketio.on('static-linear-request')
def getLinearDialogue(retried=0):
    file_path = f'../data/transcript_data/transcript-{processed_file_suffix}'
    try:
        with open(file_path, 'r', encoding='utf8') as file:
            payload = json.load(file)
        socketio.emit('linear-response', payload)
        retried = 0
    except FileNotFoundError:
        if retried==2:
            socketio.emit('unable-to-open', file_path)
            return
        process_uploaded_transcript(uploaded_file_path, processed_file_suffix)
        getLinearDialogue(retried=(retried+1))

@socketio.on('static-diagram-request')
def getDiagramData(query='', retried=0):
    file_path = f'../data/diagram_data/diagram-{processed_file_suffix}'
    try:
        with open(file_path, 'r', encoding='utf8') as file:
            data = json.load(file)
            
            emission_ID = f'diagram-response-all'
        socketio.emit(emission_ID, data)
        retried = 0
    except FileNotFoundError:
        if retried == 2:
            socketio.emit('unable-to-open', file_path)
            return
        process_uploaded_transcript(uploaded_file_path, processed_file_suffix)
        getDiagramData(query, retried=(retried+1))


@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

        
def realtime_ROS_emission():
    transcript_path = f'../data/transcript_data/transcript-{processed_file_suffix}'
    diagram_path = f'../data/diagram_data/diagram-{processed_file_suffix}'
    transcript_last_idx = -1
    while True:
        try:
            with open(transcript_path, 'r', encoding='utf8') as transcript_file:
                transcript_payload = json.load(transcript_file)
                if (transcript_payload[-1]['idx'] != transcript_last_idx):
                    print(transcript_payload[-1]['idx'])
                    print('updating...')
                    transcript_last_idx = transcript_payload[-1]['idx']
                    # print(transcript_payload)
                    socketio.emit('linear-response', transcript_payload)
            
                    diagram_payload = generate_diagram_from_file(transcript_payload, diagram_path)
                    socketio.emit('diagram-response-all', diagram_payload)
        except Exception as e:
            print(f'Exception on emission: {e}')
            socketio.emit('fatal-emission', e)
            break
        socketio.sleep(2) 


if __name__ == "__main__":
    # Start the WebSocket server
    socketio.start_background_task(realtime_ROS_emission)
    socketio.run(app, debug=False, port=6400)
