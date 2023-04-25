import json
import os
import re
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

class NERRE_Dataset(Dataset):
    def __init__(self, sentences, subject_labels, object_labels, relation_labels, entity_positions):
        self.sentences = sentences
        self.subject_labels = subject_labels
        self.object_labels = object_labels
        self.relation_labels = relation_labels
        self.entity_positions = entity_positions

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.subject_labels[idx], self.object_labels[idx], self.relation_labels[idx], self.entity_positions[idx]

def process_directory(dir_path, subject_label2idx, object_label2idx, re_label2idx):
    processed_data = []
    pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')

    for file_name in os.listdir(dir_path):
        if file_name.endswith('.json'):
            with open(os.path.join(dir_path, file_name), 'r') as file:
                content = file.read()
                data = json.loads(content)
                text = data['text']
                entities = data['entities']
                relations = data['relation_info']

            for relation in relations:
                subject_text = relation['subjectText']
                object_text = relation['objectText']
                if subject_text in text and object_text in text:
                    sentences = pattern.split(text)
                    for sentence in sentences:
                        if subject_text in sentence and object_text in sentence:
                            subject_label = subject_label2idx[relation['subjectType']]
                            object_label = object_label2idx[relation['objectType']]
                            re_label = re_label2idx[relation['rel_name']]
                            entity_positions = (text.find(subject_text), text.find(object_text))
                            processed_data.append((sentence, subject_label, object_label, re_label, entity_positions))

    return processed_data

def tokenize_data(dataset):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokenized_data = []

    for sentence, subject_label, object_label, relation_label, entity_positions in dataset:
        encoding = tokenizer(sentence, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        tokenized_data.append((input_ids, attention_mask, subject_label, object_label, relation_label, entity_positions))

    return tokenized_data

def create_data_loader(tokenized_data, batch_size):
    data_loader = DataLoader(tokenized_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return data_loader

def collate_fn(batch):
    input_ids, attention_mask, subject_labels, object_labels, relation_labels, entity_positions = zip(*batch)
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    subject_labels = torch.tensor(subject_labels, dtype=torch.long)
    object_labels = torch.tensor(object_labels, dtype=torch.long)
    relation_labels = torch.tensor(relation_labels, dtype=torch.long)
    return input_ids, attention_mask, subject_labels, object_labels, relation_labels, entity_positions
