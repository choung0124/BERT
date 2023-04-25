import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForTokenClassification

# Set the directory containing the preprocessed data
ner_data_dir = "training_data"

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
ner_model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Set the hyperparameters for fine-tuning
num_epochs = 10
batch_size = 16
learning_rate = 2e-5

# Create a label-to-ID mapping
label_to_id = {}

for file_name in os.listdir(ner_data_dir):
    if file_name.endswith("_ner_data.txt"):
        with open(os.path.join(ner_data_dir, file_name), "r") as f:
            lines = f.readlines()
            labels = []
            for line in lines:
                try:
                    label = line.split()[1]
                    labels.append(label)
                except IndexError:
                    pass  # Ignore the line if it doesn't have at least two elements after splitting
            label_to_id.update({label: idx for idx, label in enumerate(set(labels))})

# Tokenize the NER data and generate labels
ner_input_ids = []
ner_attention_masks = []
ner_labels = []
all_labels = []

for file_name in os.listdir(ner_data_dir):
    if file_name.endswith("_ner_data.txt"):
        with open(os.path.join(ner_data_dir, file_name), "r") as f:
            lines = f.readlines()
            tokens = [line.split()[0] for line in lines if len(line.split()) > 1]
            labels = []
            for line in lines:
                try:
                    label = line.split()[1]
                    labels.append(label)
                except IndexError:
                    pass  # Ignore the line if it doesn't have at least two elements after splitting
            all_labels.extend(labels)
            # Trim the tokens, attention masks, and labels to the same length
            tokens = tokens[:128]
            labels = labels[:128]
            encoded = tokenizer.encode_plus(tokens, add_special_tokens=True, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
            ner_input_ids.append(encoded["input_ids"])
            ner_attention_masks.append(encoded["attention_mask"])
            ner_labels.append(torch.tensor([label_to_id[label] for label in labels if label in label_to_id]))

ner_input_ids = torch.cat(ner_input_ids, dim=0)
ner_attention_masks = torch.cat(ner_attention_masks, dim=0)
ner_labels = torch.cat(ner_labels, dim=0)

# Create a DataLoader for the NER data
ner_dataset = TensorDataset(ner_input_ids, ner_attention_masks, ner_labels)
ner_loader = DataLoader(ner_dataset, batch_size=batch_size)

# Fine-tune the BERT NER model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
