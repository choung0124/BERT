import torch
import os
import sys
import argparse
import pickle
from transformers import BertTokenizerFast, BertModel
from model_definition import NER_RE_Model

def load_model(model_dir):
    # Load the pretrained BERT model
    bert = BertModel.from_pretrained(model_dir)

    # Load the custom head
    checkpoint = torch.load(os.path.join(model_dir, 'custom_head.pt'))
    ner_classifier_state_dict = checkpoint['ner_classifier']
    re_classifier_state_dict = checkpoint['re_classifier']

    # Load the label dictionaries
    with open(os.path.join(model_dir, 'label_dicts.pkl'), 'rb') as f:
        label_dicts = pickle.load(f)
        ner_label2idx, ner_idx2label = label_dicts['ner']
        re_label2idx, re_idx2label = label_dicts['re']

    # Initialize the model
    model = NER_RE_Model(len(ner_label2idx), len(re_label2idx), ner_label2idx, re_label2idx)
    model.bert = bert
    model.ner_classifier.load_state_dict(ner_classifier_state_dict)
    model.re_classifier.load_state_dict(re_classifier_state_dict)

    return model, ner_idx2label, re_idx2label

def predict_relations(model, tokenizer, text):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        ner_logits, re_logits = model(input_ids, attention_mask)

        ner_predictions = torch.argmax(ner_logits, dim=-1).squeeze().tolist()
        re_predictions = torch.argmax(re_logits, dim=-1).squeeze().tolist()

    entities = []
    relations = []

    for i, ner_pred in enumerate(ner_predictions):
        if ner_pred != 0:  # Ignore 'O' label
            entity = (tokenizer.convert_ids_to_tokens(input_ids[0][i]), ner_idx2label[ner_pred])
            entities.append(entity)

    for i, re_pred in enumerate(re_predictions):
        if re_pred != 0:  # Ignore 'no_relation' label
            subject = entities[i]
            object = entities[i+1]
            relation = re_idx2label[re_pred]
            relations.append((subject, object, relation))

    return relations

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_text', type=str, required=True, help='Input text to extract relations from')
    args = parser.parse_args()

    model_dir = "model"
    model, ner_idx2label, re_idx2label = load_model(model_dir)

    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    text = args.input_text

    relations = predict_relations(model, tokenizer, text)
    print("Extracted relations:", relations)
