import json
import os
import re
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
import torch.nn.functional as F

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
    # Initialize lists to store the results
    processed_data = []

    # Define a regular expression pattern to match sentence boundaries
    pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')

    # Loop through each file in the directory
    for file_name in os.listdir(dir_path):
        if file_name.endswith('.json'):
            # Read the JSON file and extract the content field from each object
            with open(os.path.join(dir_path, file_name), 'r') as file:
                content = file.read()
                data = json.loads(content)
                text = data['text']
                entities = data['entities']
                relations = data['relation_info']

            # Create a dictionary for easy access to entity types
            entity_type_dict = {entity['entityId']: entity['entityType'] for entity in entities}

            # Find the sentence containing each relation
            for relation in relations:
                subject_text = relation['subjectText']
                object_text = relation['objectText']
                if subject_text in text and object_text in text:
                    sentences = pattern.split(text)
                    for sentence in sentences:
                        if subject_text in sentence and object_text in sentence:
                            subject_type = entity_type_dict[relation['subjectID']]
                            object_type = entity_type_dict[relation['objectId']]
                            subject_label = subject_label2idx[subject_type]
                            object_label = object_label2idx[object_type]
                            re_label = re_label2idx[relation['rel_name']]
                            entity_positions = (text.find(subject_text), text.find(object_text))
                            processed_data.append((sentence, subject_label, object_label, re_label, entity_positions))

    return processed_data


def tokenize_our_data(dataset):
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
    batch_size = len(input_ids)
    max_seq_len = max(len(ids) for ids in input_ids)

    input_ids_padded = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    attention_mask_padded = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    subject_labels_padded = torch.full((batch_size, max_seq_len), -100, dtype=torch.long)
    object_labels_padded = torch.full((batch_size, max_seq_len), -100, dtype=torch.long)

    for i, ids in enumerate(input_ids):
        input_ids_padded[i, :len(ids)] = ids
        attention_mask_padded[i, :len(ids)] = 1
        subject_start, object_start = entity_positions[i]
        subject_labels_padded[i, subject_start] = subject_labels[i]
        object_labels_padded[i, object_start] = object_labels[i]

    relation_labels = torch.tensor(relation_labels, dtype=torch.long)

    return input_ids_padded, attention_mask_padded, subject_labels_padded, object_labels_padded, relation_labels, entity_positions




