import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import BertTokenizerFast
from model_definition import NER_RE_Model
from data_prep import tokenize_data, NERRE_Dataset, ner_label2idx, re_label2idx, tokenized_train_data
import os

train_data_loader = DataLoader(tokenized_train_data, batch_size=8, shuffle=True)

model = NER_RE_Model(len(ner_label2idx), len(re_label2idx))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=3e-5)

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    for batch in tqdm(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        subject_labels = batch['subject_label'].to(device)
        object_labels = batch['object_label'].to(device)
        re_labels = batch['re_label'].to(device)
        entity_positions = batch['entity_positions'].to(device)

        optimizer.zero_grad()
        subject_logits, object_logits, re_logits = model(input_ids, attention_mask, entity_positions)
        loss_fn = torch.nn.CrossEntropyLoss()
        subject_loss = loss_fn(subject_logits.view(-1, subject_logits.size(-1)), subject_labels.view(-1))
        object_loss = loss_fn(object_logits.view(-1, object_logits.size(-1)), object_labels.view(-1))
        re_loss = loss_fn(re_logits, re_labels)
        total_loss = subject_loss + object_loss + re_loss
        total_loss.backward()
        optimizer.step()

# ... (rest of the code remains the same)

num_epochs = 3

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_epoch(model, train_data_loader, optimizer, device)

model_dir = "model"
model.bert.save_pretrained(model_dir)

# Save the custom head
torch.save({
    'subject_classifier': model.subject_classifier.state_dict(),
    'object_classifier': model.object_classifier.state_dict(),
    're_classifier': model.re_classifier.state_dict()
}, os.path.join(model_dir, 'custom_head.pt'))

# Save the tokenizer
tokenizer.save_pretrained(model_dir)

# ... (rest of the code remains the same)

