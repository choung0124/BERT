import os
import torch
import json
import argparse
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForTokenClassification, BertForSequenceClassification


def extract_relationship(text, ner_model, re_model, tokenizer, label_to_id, relation_to_id):
    tokens = tokenizer.tokenize(text)
    encoded = tokenizer.encode_plus(tokens, add_special_tokens=True, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    
    # Set the confidence threshold for NER predictions
    conf_threshold = 0.5

    # Predict NER labels
    ner_model.eval()
    with torch.no_grad():
        logits = ner_model(input_ids, attention_mask=attention_mask).logits
        predictions = torch.argmax(logits, dim=-1).squeeze().tolist()
        probabilities = torch.softmax(logits, dim=-1).squeeze().tolist()

    # Extract entities
    entities = []
    current_entity = []
    current_label = None
    for i, (token, prediction, probability) in enumerate(zip(tokens, predictions, probabilities)):
        if prediction != -100 and probability[prediction] > conf_threshold and prediction in label_to_id.values():
            label = list(label_to_id.keys())[list(label_to_id.values()).index(prediction)]
            if label.startswith("B-"):
                if current_entity:
                    entities.append((current_label, " ".join(current_entity)))
                current_entity = [token]
                current_label = label[2:]
            elif label.startswith("I-") and current_label == label[2:]:
                current_entity.append(token)
        else:
            if current_entity:
                entities.append((current_label, " ".join(current_entity)))
                current_entity = []
                current_label = None

    # Extract relationships
    relationships = []
    for i, (subject_label, subject) in enumerate(entities):
        for j, (object_label, obj) in enumerate(entities):
            if i != j:
                tokens = tokenizer.tokenize(f"{subject} [SEP] {obj}")
                encoded = tokenizer.encode_plus(tokens, add_special_tokens=True, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]

                # Predict relationship
                re_model.eval()
                with torch.no_grad():
                    logits = re_model(input_ids, attention_mask=attention_mask).logits
                    predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

                # Add predicted relationship to the list
                if predictions in relation_to_id.values():
                    relation = list(relation_to_id.keys())[list(relation_to_id.values()).index(predictions)]
                    relationships.append((subject_label, subject, relation, object_label, obj))

    return relationships


def load_mapping_from_json(file_path):
    with open(file_path, "r") as f:
        mapping = json.load(f)
    return mapping

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract relationships from a given input text using fine-tuned NER and RE models.")
    parser.add_argument("input_text", help="Input text to extract relationships from.")
    parser.add_argument("--label_to_id_file", default="label_to_id.json", help="Path to the JSON file containing label-to-ID mapping for NER.")
    parser.add_argument("--relation_to_id_file", default="relation_to_id.json", help="Path to the JSON file containing relation-to-ID mapping for RE.")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="Threshold for confidence of NER predictions to consider as valid.")
    args = parser.parse_args()

    # Load the input text
    input_text = args.input_text

    # Load the fine-tuned NER model and tokenizer
    ner_output_dir = "models/ner"
    ner_model = BertForTokenClassification.from_pretrained(ner_output_dir)
    tokenizer = BertTokenizer.from_pretrained(ner_output_dir)

    # Load the fine-tuned RE model
    re_output_dir = "models/re"
    re_model = BertForSequenceClassification.from_pretrained(re_output_dir)

    # Specify label-to-ID mapping for NER
    label_to_id = load_mapping_from_json(args.label_to_id_file)

    # Specify relation-to-ID mapping for RE
    relation_to_id = load_mapping_from_json(args.relation_to_id_file)

    # Extract relationships
    conf_threshold = args.conf_threshold
    relationships = extract_relationship(input_text, ner_model, re_model, tokenizer, label_to_id, relation_to_id, conf_threshold)

    # Print the extracted relationships
    if relationships:
        print("Extracted relationships:")
        for relationship in relationships:
            subject_label, subject, relation, object_label, obj = relationship
            print(f"{subject} [{subject_label}] --{relation}--> {obj} [{object_label}]")
    else:
        print("No relationships found.")

