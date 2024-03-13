from flask import Flask, jsonify, request
from flask_cors import CORS
import json

CHAT_PATH = "data/prepped_for_chats/chat-log-03-12.json"
DIAGRAM_PATH = "data/prepped_for_diagrams/diagram-log-03-12.json"

app = Flask(__name__)
CORS(app)

@app.route("/api/uploadTranscript", methods=["POST"])
def process_uploaded_transcript():
    """
    Saves and processes the transcript uploaded 
    by the user of the app and processes it for visualization
    """
    pass


@app.route("/api/health", methods=["GET"])
def healthchecker():
    """
    Standard health checker for the server
    """
    return {"status": "success"}


@app.route("/api/chatjson", methods=["GET"])
def return_raw_json():
    """
    returns a json file containing the sequential lines of dialogue and their relevant data in a single json object array
    """
    with open(CHAT_PATH, "r", encoding="utf8") as file:
        payload = json.load(file)
        return jsonify(payload)


@app.route("/api/diagramjson", methods=["GET"])
def return_diagram_json():
    """
    Returns a json file processed to contain a list of nodes and links for the diagrams to use in their rendering
    """
    selection = request.args.get('selection')
    with open(DIAGRAM_PATH, "r", encoding="utf8") as file:
        json_data = json.load(file)
        
        if selection in 'nodes':
            return jsonify(json_data['nodes'])
        else:
            return jsonify(json_data)
        


if __name__ == "__main__":
    app.run(port=6400)
