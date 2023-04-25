import argparse
import torch
from transformers import BertTokenizerFast
from model_definition import NER_RE_Model
import json

# Load the trained model
model_dir = "model"
tokenizer = BertTokenizerFast.from_pretrained(model_dir)
model = NER_RE_Model.from_pretrained(model_dir)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Define command line arguments
parser = argparse.ArgumentParser(description='Extract relationships from text')
parser.add_argument('text', type=str, help='Input text')

# Parse command line arguments
args = parser.parse_args()
text = args.text

# Tokenize input text
encoding = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
input_ids = encoding['input_ids'].to(device)
attention_mask = encoding['attention_mask'].to(device)

# Make predictions on input text
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
# Extract subject, object and relationship labels from predictions
ner_logits = outputs['ner_logits'].argmax(dim=-1).cpu().numpy().tolist()[0][1:-1]
ner_labels = [model.ner_idx2label[label_idx] for label_idx in ner_logits]
re_logits = outputs['re_logits'].argmax(dim=-1).cpu().numpy().tolist()[0][1:-1]
re_labels = [model.re_idx2label[label_idx] for label_idx in re_logits]

# Extract relationship tuples
relations = []
subject_text = None
object_text = None
for i, (ner_label, re_label) in enumerate(zip(ner_labels, re_labels)):
    if ner_label.startswith('B-'):
        # Start of a new entity
        if ner_label[2:] == 'SUB':
            subject_text = tokenizer.decode(input_ids[0, i+1].unsqueeze(dim=0).cpu().numpy().tolist()[0])
        elif ner_label[2:] == 'OBJ':
            object_text = tokenizer.decode(input_ids[0, i+1].unsqueeze(dim=0).cpu().numpy().tolist()[0])
    elif ner_label.startswith('I-'):
        # Continuation of the current entity
        continue
    else:
        # End of an entity
        if subject_text is not None and object_text is not None:
            relation_tuple = {'subject': subject_text, 'object': object_text, 'relation': re_label}
            relations.append(relation_tuple)
        subject_text = None
        object_text = None

# Print out the extracted relationships
if len(relations) == 0:
    print("No relationships found in the input text")
else:
    print(json.dumps(relations, indent=4))
