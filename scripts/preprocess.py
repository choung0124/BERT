import json
import os

def preprocess_ner(json_data):
    ner_data = []
    
    for entity in json_data["entities"]:
        ner_data.append((entity["span"]["begin"], entity["span"]["end"], entity["entityType"]))
    
    ner_data.sort(key=lambda x: x[0])
    
    text = json_data["text"]
    ner_tags = []
    current_idx = 0
    
    for begin, end, entity_type in ner_data:
        while current_idx < begin:
            ner_tags.append((text[current_idx], "O"))
            current_idx += 1
        
        first = True
        for i in range(begin, end):
            ner_tags.append((text[i], f"B-{entity_type}" if first else f"I-{entity_type}"))
            first = False
            current_idx += 1
    
    while current_idx < len(text):
        ner_tags.append((text[current_idx], "O"))
        current_idx += 1
    
    return ner_tags


def preprocess_re(json_data):
    re_data = []
    
    for relation in json_data["relation_info"]:
        re_data.append((relation["subjectText"], relation["objectText"], relation["rel_name"]))
    
    return re_data

# Set the directory containing the JSON files
json_directory = "test"

# Set the directory to save the preprocessed data
output_directory = "training_data"

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Iterate through all JSON files in the directory
for file_name in os.listdir(json_directory):
    if file_name.endswith(".json"):
        json_path = os.path.join(json_directory, file_name)

        # Load the JSON data
        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)

        # Preprocess the data for NER and RE tasks
        ner_data = preprocess_ner(json_data)
        re_data = preprocess_re(json_data)

        # Save the preprocessed data to files
        base_name = os.path.splitext(file_name)[0]
        ner_file_name = f"{base_name}_ner_data.txt"
        re_file_name = f"{base_name}_re_data.txt"
        ner_file_path = os.path.join(output_directory, ner_file_name)
        re_file_path = os.path.join(output_directory, re_file_name)

        with open(ner_file_path, "w") as ner_file:
            for token, tag in ner_data:
                ner_file.write(f"{token} {tag}\n")

        with open(re_file_path, "w") as re_file:
            for subject, relation, obj in re_data:
                re_file.write(f"{subject} {relation} {obj}\n")
