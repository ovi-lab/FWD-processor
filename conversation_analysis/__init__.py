from conversation_analysis import app

if __name__ == "__main__":
    # Start the WebSocket server
    app.socketio.run(app.app, debug=False, port=6400, use_reloader=False, log_output=False)