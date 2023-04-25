import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForTokenClassification

# Set the directory containing the preprocessed data
ner_data_dir = "training_data"

# Load the pre-trained BERT model and tokenizer
num_labels = len(set(all_labels))
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
ner_model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

# Set the hyperparameters for fine-tuning
num_epochs = 10
batch_size = 16
learning_rate = 2e-5

# Tokenize the NER data and generate labels
ner_input_ids = []
ner_attention_masks = []
ner_labels = []
all_labels = []

# Read all labels from the preprocessed data
for file_name in os.listdir(ner_data_dir):
    if file_name.endswith("_ner_data.txt"):
        with open(os.path.join(ner_data_dir, file_name), "r") as f:
            lines = f.readlines()
            labels = [line.split()[1] for line in lines if len(line.split()) > 1]
            all_labels.extend(labels)

# Create a label-to-ID mapping
label_to_id = {label: idx for idx, label in enumerate(set(all_labels))}

# Tokenize and align the labels
for file_name in os.listdir(ner_data_dir):
    if file_name.endswith("_ner_data.txt"):
        with open(os.path.join(ner_data_dir, file_name), "r") as f:
            lines = f.readlines()
            tokens = []
            labels = []
            for line in lines:
                if len(line.split()) > 1:
                    token = line.split()[0]
                    label = line.split()[1]
                    labels.append(label)
                    sub_tokens = tokenizer.tokenize(token)
                    tokens.extend(sub_tokens)
            encoded = tokenizer.encode_plus(tokens, add_special_tokens=True, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
            ner_input_ids.append(encoded["input_ids"])
            ner_attention_masks.append(encoded["attention_mask"])
            
            aligned_labels = []
            for label in labels:
                if label in label_to_id:
                    label_id = label_to_id[label]
                    sub_tokens = tokenizer.tokenize(label.split()[0])
                    aligned_labels.extend([label_id] + [-100] * (len(sub_tokens) - 1))
            padded_labels = aligned_labels[:512]  # Truncate to match the max_length
            padded_labels.extend([-100] * (512 - len(padded_labels)))  # Pad to match the max_length
            ner_labels.append(torch.tensor(padded_labels))

ner_input_ids = torch.cat(ner_input_ids, dim=0)
ner_attention_masks = torch.cat(ner_attention_masks, dim=0)
ner_labels = torch.stack(ner_labels, dim=0)

print(f"ner_input_ids shape: {ner_input_ids.shape}")
print(f"ner_attention_masks shape: {ner_attention_masks.shape}")
print(f"ner_labels shape: {ner_labels.shape}")

# Create a DataLoader for the NER data
ner_dataset = TensorDataset(ner_input_ids, ner_attention_masks, ner_labels)
ner_loader = DataLoader(ner_dataset, batch_size=batch_size)

# Fine-tune the BERT NER model
device = torch.device("cpu")
ner_model.to(device)

optimizer = torch.optim.AdamW(ner_model.parameters(), lr=learning_rate)
total_steps = len(ner_loader) * num_epochs

for epoch in range(num_epochs):
    for batch in ner_loader:
        input_ids, attention_masks, labels = tuple(t.to(device) for t in batch)
        outputs = ner_model(input_ids, attention_mask=attention_masks, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Save the fine-tuned BERT NER model
output_dir = "models/ner"
os.makedirs(output_dir, exist_ok=True)
ner_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
