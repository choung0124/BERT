import sys
import json
import torch
from transformers import BertTokenizer, BertForTokenClassification, BertForSequenceClassification, pipeline
from itertools import groupby


def extract_entities(text, ner_model, tokenizer, id_to_label):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = ner_model(input_ids, attention_mask=attention_mask)
    predictions = torch.argmax(outputs.logits, dim=2)

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    tags = [id_to_label[prediction.item()] for prediction in predictions.squeeze()]

    entities = []
    for tag, group in groupby(zip(tokens, tags), lambda x: x[1]):
        if tag != "O":
            entity = "".join([t[2:] if t.startswith("##") else t for t, _ in group]).replace("[SEP]", "").replace("[CLS]", "").strip()
            entities.append((tag, entity))
    return entities


def extract_relationships(entities, re_model, tokenizer, id_to_relation):
    relationships = []

    for i, (tag1, entity1) in enumerate(entities):
        for j, (tag2, entity2) in enumerate(entities):
            if i != j:
                inputs = tokenizer(f"{entity1} [SEP] {entity2}", return_tensors="pt")
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]

                with torch.no_grad():
                    outputs = re_model(input_ids, attention_mask=attention_mask)
                prediction = torch.argmax(outputs.logits, dim=1).item()

                relation = id_to_relation[prediction]
                if relation != "no_relation":
                    relationships.append((entity1, relation, entity2))
    return relationships


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_models.py <input_text>")
        sys.exit(1)

    input_text = sys.argv[1]

    # Load the trained NER and RE models
    ner_model = pipeline('ner', model='models/ner', tokenizer='bert-base-cased')
    re_model = BertForSequenceClassification.from_pretrained("models/re")
    tokenizer = BertTokenizer.from_pretrained("models/ner")

    # Load the label mappings
    with open("label_to_id.json", "r") as f:
        label_to_id = json.load(f)
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    with open("relation_to_id.json", "r") as f:
        relation_to_id = json.load(f)
    id_to_relation = {idx: relation for relation, idx in relation_to_id.items()}

    # Extract entities and relationships from the input text
    entities = extract_entities(input_text, ner_model, tokenizer, id_to_label)
    relationships = extract_relationships(entities, re_model, tokenizer, id_to_relation)

    # Print the extracted relationships
    print("Extracted relationships:")
    for subject, relationship, obj in relationships:
        print(f"{subject} - {relationship} - {obj}")
