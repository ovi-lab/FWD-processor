from flask import Flask, jsonify, request
from flask_cors import CORS
import json

SEQUENTIAL_PATH = "data/log-02-20.json"
DIAGRAM_PATH = "output/new_robot_data_2_20.json"

app = Flask(__name__)
CORS(app)


@app.route("/api/health", methods=["GET"])
def healthchecker():
    return {"status": "success"}


@app.route("/api/rawjson", methods=["GET"])
def return_static_json():
    with open(SEQUENTIAL_PATH, "r", encoding="utf8") as file:
        payload = json.load(file)
        return jsonify(payload)


@app.route("/api/diagramFormat", methods=["GET"])
def return_diagram_json():
    selection = request.args.get('selection')
    with open(DIAGRAM_PATH, "r", encoding="utf8") as file:
        json_data = json.load(file)
        if selection == 'nodes':
            payload = json_data['nodes']
        return jsonify(payload)


if __name__ == "__main__":
    app.run(port=6400)
