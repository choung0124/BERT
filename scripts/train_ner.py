import json
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForTokenClassification
import warnings
from transformers import logging

logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.modeling_utils")

def preprocess_ner(json_data):
    ner_data = []
    
    for entity in json_data["entities"]:
        begin = entity["span"]["begin"]
        end = entity["span"]["end"]
        entity_type = entity["entityType"]
        ner_data.append((begin, end, entity_type))
        print(f"entity: {entity}, ner_data[-1]: {ner_data[-1]}")
    
    ner_data.sort(key=lambda x: x[0])
    
    text = json_data["text"]
    ner_tags = []
    current_idx = 0
    
    for begin, end, entity_type in ner_data:
        while current_idx < begin:
            ner_tags.append((text[current_idx], "O"))
            current_idx += 1
        
        first = True
        for i in range(begin, end):
            ner_tags.append((text[i], f"B-{entity_type}" if first else f"I-{entity_type}"))
            first = False
            current_idx += 1
    
        current_idx = end
    
    while current_idx < len(text):
        ner_tags.append((text[current_idx], "O"))
        current_idx += 1
    
    return ner_tags


# Set the directory containing the JSON files
json_directory = "test"

# Preprocessed data
preprocessed_data = []

# Iterate through all JSON files in the directory
for file_name in os.listdir(json_directory):
    if file_name.endswith(".json"):
        json_path = os.path.join(json_directory, file_name)

        # Load the JSON data
        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)

        # Preprocess the data for NER tasks
        ner_data = preprocess_ner(json_data)
        preprocessed_data.append(ner_data)
        print(f"Processed: {file_name}")
        print(f"Number of entities: {len(json_data['entities'])}")
        for entity in json_data['entities']:
            print(entity)

# Hyperparameters
num_epochs = 10
batch_size = 8
learning_rate = 2e-5

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Initialize the BERT model for token classification
all_labels = [tag for ner_data in preprocessed_data for _, tag in ner_data]
num_unique_labels = len(set(all_labels))
ner_model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=num_unique_labels)

# Preprocess and tokenize the NER data
ner_input_ids, ner_attention_masks, ner_labels = [], [], []
label_to_id = {label: idx for idx, label in enumerate(set(all_labels))}

# Tokenize and align the labels
for ner_data in tqdm(preprocessed_data, desc="Tokenizing and aligning labels"):
    tokens, labels = [], []
    for begin, end, entity_type in ner_data:
        entity_tokens = tokenizer.tokenize(json_data["text"][begin:end])
        entity_labels = [label_to_id[f"B-{entity_type}"]] + [label_to_id[f"I-{entity_type}"]] * (len(entity_tokens) - 1)
        tokens.extend(entity_tokens)
        labels.extend(entity_labels)
    
    encoded = tokenizer.encode_plus(tokens, add_special_tokens=True, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    ner_input_ids.append(encoded["input_ids"])
    ner_attention_masks.append(encoded["attention_mask"])
    aligned_labels = []
    for label in labels:
        label_id = label_to_id[label]
        sub_tokens = tokenizer.tokenize(label.split()[0])
        aligned_labels.extend([label_id] + [-100] * (len(sub_tokens) - 1))

    padded_labels = aligned_labels[:512]
    padded_labels.extend([-100] * (512 - len(padded_labels)))
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ner_model.to(device)

optimizer = torch.optim.AdamW(ner_model.parameters(), lr=learning_rate)
total_steps = len(ner_loader) * num_epochs

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print("-" * 40)

    ner_model.train()
    epoch_loss = 0
    num_batches = 0

    # Add a progress bar for the batches
    for batch in tqdm(ner_loader, desc="Training", unit="batch"):
        input_ids, attention_masks, labels = tuple(t.to(device) for t in batch)
        outputs = ner_model(input_ids, attention_mask=attention_masks, labels=labels)
        loss = outputs.loss
        epoch_loss += loss.item()
        num_batches += 1

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_epoch_loss = epoch_loss / num_batches
    print(f"Average training loss: {avg_epoch_loss:.4f}")

# Save the fine-tuned BERT NER model
output_dir = "models/ner"
os.makedirs(output_dir, exist_ok=True)
ner_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

mapping_file = "label_to_id.json"

with open(mapping_file, "w") as f:
    json.dump(label_to_id, f)

