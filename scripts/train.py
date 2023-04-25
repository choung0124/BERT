import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import BertTokenizerFast
from model_definition import NER_RE_Model
from data_prep import tokenize_our_data, create_data_loader, NERRE_Dataset, process_directory
import os
import pickle

from generate_label_dicts import generate_label_dicts

def train_epoch(model, data_loader, optimizer, device):
    model = model.train()

    # Initialize train_loss
    train_loss = 0

    for batch in data_loader:
        input_ids, attention_mask, subject_labels, object_labels, relation_labels, _ = batch

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        subject_labels = subject_labels.to(device)
        object_labels = object_labels.to(device)
        relation_labels = relation_labels.to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Reshape logits and labels tensors
        ner_logits = outputs['ner_logits'].view(-1, outputs['ner_logits'].shape[-1])
        subject_labels = subject_labels.view(-1)
        object_labels = object_labels.view(-1)

        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        subject_loss = criterion(ner_logits, subject_labels)
        object_loss = criterion(ner_logits, object_labels)

        loss = subject_loss + object_loss

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Return the average train_loss
    return train_loss / len(data_loader)


dir_path = "test"
subject_label2idx, object_label2idx, re_label2idx, ner_label2idx = generate_label_dicts(dir_path)

filtered_data = process_directory(dir_path, subject_label2idx, object_label2idx, re_label2idx)
train_dataset = NERRE_Dataset(*zip(*filtered_data))

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
tokenized_train_data = tokenize_our_data(train_dataset)
train_data_loader = DataLoader(tokenized_train_data, batch_size=8, shuffle=True)

model = NER_RE_Model(len(ner_label2idx), len(re_label2idx), ner_label2idx, re_label2idx)

batch_size = 8
train_data_loader = create_data_loader(tokenized_train_data, batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=3e-5)

num_epochs = 3

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_epoch(model, train_data_loader, optimizer, device)

model_dir = "model"
model.bert.save_pretrained(model_dir)

# Save the custom head
torch.save({
    'ner_classifier': model.ner_classifier.state_dict(),
    're_classifier': model.re_classifier.state_dict()
}, os.path.join(model_dir, 'custom_head.pt'))

# Save the tokenizer
tokenizer.save_pretrained(model_dir)

# Create reverse label dictionaries
ner_idx2label = {v: k for k, v in ner_label2idx.items()}
re_idx2label = {v: k for k, v in re_label2idx.items()}

# Save the label dictionaries
label_dicts = {
    'ner': (ner_label2idx, ner_idx2label),
    're': (re_label2idx, re_idx2label)
}

with open(os.path.join(model_dir, 'label_dicts.pkl'), 'wb') as f:
    pickle.dump(label_dicts, f)
