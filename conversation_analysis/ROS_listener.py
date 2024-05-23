import roslibpy
client = roslibpy.Ros(host='localhost', port=9090)
client.run()

def callback(data):
    # Process the received data here
    print(data)
    pass

listener = roslibpy.Topic(client, '/strawberry/dialog_result', 'std_msgs/String')
listener.subscribe(callback)