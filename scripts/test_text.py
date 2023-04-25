import argparse
import torch
from transformers import BertTokenizerFast
from model_definition import NER_RE_Model

parser = argparse.ArgumentParser()
parser.add_argument('--text', type=str, required=True, help='Text to extract relationships from')
parser.add_argument('--model_dir', type=str, default='model', help='Directory where the model files are saved')
args = parser.parse_args()

tokenizer = BertTokenizerFast.from_pretrained(args.model_dir)
model = NER_RE_Model(num_labels_ner, num_labels_re, ner_label2idx, re_label2idx)

# Load the pre-trained weights
model.load_state_dict(torch.load(f'{args.model_dir}/model_weights.pt'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

text = args.text
encoding = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

input_ids = encoding['input_ids'].to(device)
attention_mask = encoding['attention_mask'].to(device)

with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

ner_labels = outputs['ner_labels'][0].cpu().numpy()
re_labels = outputs['re_labels'][0].cpu().numpy()

ner_idx2label = {v: k for k, v in ner_label2idx.items()}
re_idx2label = {v: k for k, v in re_label2idx.items()}

subject_entities = []
object_entities = []
relationships = []

# Loop through each relationship and extract the subject entity, object entity, and relationship type
for i, re_label in enumerate(re_labels):
    if re_idx2label[re_label] != 'no_relation':
        subject_start, subject_end, object_start, object_end = entity_positions[i]
        subject_entities.append(text[subject_start:subject_end])
        object_entities.append(text[object_start:object_end])
        relationships.append(re_idx2label[re_label])

# Print the extracted relationships
for subject_entity, relationship, object_entity in zip(subject_entities, relationships, object_entities):
    print(f'{subject_entity} -- {relationship} --> {object_entity}')
