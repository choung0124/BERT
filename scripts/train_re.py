import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification

# Set the directory containing the preprocessed data
re_data_dir = "training_data"

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
re_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Set the hyperparameters for fine-tuning
num_epochs = 10
batch_size = 16
learning_rate = 2e-5

# Tokenize the RE data and create a training set
re_input_ids = []
re_attention_masks = []
re_labels = []

# Create a mapping from relation labels to their respective IDs
relation_to_id = {}

# Open and read the training data file
train_data_file = os.path.join(re_data_dir, "train_data.txt")
with open(train_data_file, "r") as f:
    lines = f.readlines()

# Loop through each line of the training data
for line in lines:
    # Split the line into subject, object, relation
    subject, obj, relation = line.strip().split("\t")
    
    # Add the relation to the dictionary if it doesn't exist yet
    if relation not in relation_to_id:
        relation_to_id[relation] = len(relation_to_id)

# Tokenize and align the relations
def extract_subject_relation_object(line):
    words = line.split()
    relation = words[-1]
    obj = words[-2]
    subject = ' '.join(words[:-2])
    return subject, relation, obj

for file_name in os.listdir(re_data_dir):
    if file_name.endswith("_re_data.txt"):
        with open(os.path.join(re_data_dir, file_name), "r") as f:
            line = f.readline()
            subject, relation, obj = extract_subject_relation_object(line)

            tokens = tokenizer.tokenize(f"{subject} [SEP] {obj}")
            encoded = tokenizer.encode_plus(tokens, add_special_tokens=True, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
            re_input_ids.append(encoded["input_ids"])
            if relation not in relation_to_id:
                print(f"Error: Relation '{relation}' not found in the relation_to_id dictionary")
                continue

            re_labels.append(torch.tensor(relation_to_id[relation]))


re_input_ids = torch.cat(re_input_ids, dim=0)
re_attention_masks = torch.cat(re_attention_masks, dim=0)
re_labels = torch.cat(re_labels, dim=0)

# Create a DataLoader for the RE data
re_dataset = TensorDataset(re_input_ids, re_attention_masks, re_labels)
re_loader = DataLoader(re_dataset, batch_size=batch_size)

# Fine-tune the BERT RE model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
re_model.to(device)

optimizer = torch.optim.AdamW(re_model.parameters(), lr=learning_rate)
total_steps = len(re_loader) * num_epochs

for epoch in range(num_epochs):
    for batch in re_loader:
        input_ids, attention_masks, labels = tuple(t.to(device) for t in batch)
        outputs = re_model(input_ids, attention_mask=attention_masks, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Save the fine-tuned BERT RE model
output_dir = "models/re"
os.makedirs(output_dir, exist_ok=True)
re_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
