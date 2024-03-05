from flask import Flask, jsonify
from flask_cors import CORS
import json


app = Flask(__name__)
CORS(app)

@app.route("/api/health", methods=["GET"])
def healthchecker():
    return {"status": "success"}


@app.route("/api/rawjson", methods=["GET"])
def return_static_json():
    with open("data/log-02-20.json", "r", encoding="utf8") as file:
        json_file = json.load(file)

        return jsonify(json_file)


if __name__ == "__main__":
    app.run(port=6400)
