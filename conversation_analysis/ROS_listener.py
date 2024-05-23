import roslibpy
client = roslibpy.Ros(host='10.80.58.230', port=9090)
client.run()

def callback(data):
    # Process the received data here
    print(data)
    pass

listener = roslibpy.Topic(client, '/strawberry/dialog_result', 'strawberry_ros_msgs/DialogResult')
listener.subscribe(callback)