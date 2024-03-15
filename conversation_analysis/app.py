import os
from flask import Flask, flash
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO, emit, send
import json
import time
from process_static_transcript import process_static_transcript

CHAT_PATH = "data/prepped_for_chats/chat-log-03-12.json"
DIAGRAM_PATH = "data/prepped_for_diagrams/diagram-log-03-12.json"
processed_file_suffix = 'transcript-log.json'

data_emission = 'static'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/tmp/'
socketio = SocketIO(app, cors_allowed_origins='*')

@socketio.on('initSubscriber')
def init_subscriber():
    data_emission = 'dynamic'
    pass

@socketio.on('uploadFile')
def processIncomingFile(file_data):
    data_emission = 'static'
    print(file_data)
    
    if file_data:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'upload.yml')
        
        # Write the file data to the file
        with open(file_path, 'wb') as f:
            f.write(file_data)
        process_static_transcript(file_path, processed_file_suffix)

@socketio.on('static-linear-request')
def getLinearDialogue():
    file_path = f'data/prepped_for_chats/chat-{processed_file_suffix}'
    with open(file_path, 'r', encoding='utf8') as file:
        payload = json.load(file)
    socketio.emit('static-linear-response', payload)

@socketio.on('static-diagram-request')
def getDiagramData(query=''):
    file_path = f'data/prepped_for_diagrams/diagram-{processed_file_suffix}'
    with open(file_path, 'r', encoding='utf8') as file:
        data = json.load(file)
        if query == 'nodes' or query == 'links':
            payload = data[query]
            emission_ID = f'static-diagram-response-{query}'
        else:
            payload = data
            emission_ID = f'static-diagram-response-all'
        flash(payload)
    socketio.emit(emission_ID, payload)



# @app.route("/api/chatjson", methods=["GET"])
# def return_raw_json():
#     """
#     returns a json file containing the sequential lines of dialogue and their relevant data in a single json object array
#     """
#     with open(CHAT_PATH, "r", encoding="utf8") as file:
#         payload = json.load(file)
#         return jsonify(payload)


# @app.route("/api/diagramjson", methods=["GET"])
# def return_diagram_json():
#     """
#     Returns a json file processed to contain a list of nodes and links for the diagrams to use in their rendering
#     """
#     selection = request.args.get('selection')
#     with open(DIAGRAM_PATH, "r", encoding="utf8") as file:
#         json_data = json.load(file)
        
#         if selection in 'nodes':
#             return jsonify(json_data['nodes'])
#         else:
#             return jsonify(json_data)
        
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
        
def send_data_updates():
    while True:
        # Fetch data from your Flask application
        with open(CHAT_PATH, "r", encoding="utf8") as file:
            payload = json.load(file)
        # Send data updates to connected clients
        socketio.emit('linear_updates', payload)

        # Adjust the sleep time as needed
        socketio.sleep(1)
        


if __name__ == "__main__":
    # Start the WebSocket server
    if data_emission == 'dynamic':
        socketio.start_background_task(send_data_updates)
    socketio.run(app, debug=False, port=6400)
