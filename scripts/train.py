import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import BertTokenizerFast
from model_definition import NER_RE_Model
from data_prep import tokenize_data, create_data_loader, NERRE_Dataset, process_directory
import os

from generate_label_dicts import generate_label_dicts

dir_path = "test"
subject_label2idx, object_label2idx, re_label2idx, ner_label2idx = generate_label_dicts(dir_path)

filtered_data = process_directory(dir_path, subject_label2idx, object_label2idx, re_label2idx)
train_dataset = NERRE_Dataset(*zip(*filtered_data))

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
tokenized_train_data = tokenize_data(train_dataset)
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
