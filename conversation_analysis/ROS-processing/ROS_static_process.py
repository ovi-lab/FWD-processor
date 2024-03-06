import json
import yaml
def process_static_smalltalk_dialog(PATH):
    conversation = []
    with open(PATH, 'r', encoding='utf8') as file:
        data = yaml.safe_load_all(file)
        for doc in data:
            print(doc)
        
            
    
    

    # Write the list to a JSON file
    # file_path = 'log-03-07.json'
    # with open(file_path, 'a') as f:
    #     json.dump(conversation, f)

if __name__ == '__main__':
    process_static_smalltalk_dialog("data/dialog_result.yml")