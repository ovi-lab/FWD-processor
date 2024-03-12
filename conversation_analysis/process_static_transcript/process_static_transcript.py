from  parse_ros_output import parse_ros_output
from clean_JSON import clean_JSON_from_data
def process_static_transcript(ros_transcript, json_output):
    """
    Takes a full list of ROS nodes and converts it into JSON for use in
    """
    data = parse_ros_output(ros_transcript)
    if not data:
        print('error')
        return
    clean_JSON_from_data(data, json_output)

    

if __name__ == '__main__':
    
    process_static_transcript('data/dialog_result.yml', 'log-03-13.json')