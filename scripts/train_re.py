import os
import torch
import json
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm

def preprocess_re(json_data):
    re_data = []
    entities = {entity["entityId"]: entity for entity in json_data["entities"]}

    for relation in json_data["relation_info"]:
        subject_id, obj_id = relation["subjectID"], relation["objectId"]
        if subject_id not in entities or obj_id not in entities:
            continue
        subject = entities[subject_id]["entityName"]
        obj = entities[obj_id]["entityName"]
        re_data.append((subject, relation["rel_name"], obj))

    return re_data

# Set the hyperparameters for fine-tuning
num_epochs = 10
batch_size = 16
learning_rate = 2e-5

# Create a mapping from relation labels to their respective IDs
relations = [
    "inhibits",
    "inhibit",
    "treat",
    "treats",
    "prevent",
    "prevents",
    "reduces",
    "reduce",
    "suppress",
    "regulate",
    "stimulate",
    "associate",
    "biomarker"
]
relation_to_id = {relation: idx for idx, relation in enumerate(relations)}

# Save the relation_to_id mapping as a JSON file
with open("relation_to_id.json", "w") as f:
    json.dump(relation_to_id, f, indent=4)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
re_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(relation_to_id))

# Tokenize the RE data and create a training set
re_input_ids = []
re_attention_masks = []
re_labels = []

json_data_dir = "test"
json_files = [f for f in os.listdir(json_data_dir) if f.endswith(".json")]

for file_name in tqdm(json_files, desc="Processing JSON files"):
    with open(os.path.join(json_data_dir, file_name), "r") as f:
        json_data = json.load(f)
        re_data = preprocess_re(json_data)

        for subject, relation, obj in re_data:
            # Tokenize and encode the relation
            tokens = tokenizer.tokenize(f"{subject} [SEP] {obj}")
            encoded = tokenizer.encode_plus(tokens, add_special_tokens=True, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

            if encoded["input_ids"].shape[1] > 0:
                # Add the encoded relation and its label to the lists
                re_input_ids.append(encoded["input_ids"].squeeze(0))
                re_attention_masks.append(encoded["attention_mask"].squeeze(0))
                re_labels.append(torch.tensor(relation_to_id[relation]).unsqueeze(0))

if len(re_input_ids) > 0:
    re_input_ids = torch.stack(re_input_ids, dim=0)
    re_attention_masks = torch.stack(re_attention_masks, dim=0)
    re_labels = torch.cat(re_labels)
else:
    print("No valid inputs found. Aborting training.")
    exit()

# Create a DataLoader for the RE data
re_dataset = TensorDataset(re_input_ids, re_attention_masks, re_labels)
re_loader = DataLoader(re_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

# Fine-tune the BERT RE model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
re_model.to(device)

optimizer = torch.optim.AdamW(re_model.parameters(), lr=learning_rate)
total_steps = len(re_loader) * num_epochs

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print("-" * 40)

    re_model.train()
    epoch_loss = 0
    num_batches = 0

    # Add a progress bar for the batches
    for batch in tqdm(re_loader, desc="Training", unit="batch"):
        input_ids, attention_masks, labels = tuple(t.to(device) for t in batch)
        outputs = re_model(input_ids, attention_mask=attention_masks, labels=labels)
        loss = outputs.loss
        epoch_loss += loss.item()
        num_batches += 1

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_epoch_loss = epoch_loss / num_batches
    print(f"Average training loss: {avg_epoch_loss:.4f}")

# Save the fine-tuned BERT RE model
output_dir = "models/re"
os.makedirs(output_dir, exist_ok=True)
re_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
