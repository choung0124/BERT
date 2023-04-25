import argparse
import torch
from transformers import BertTokenizerFast
from model_definition import NER_RE_Model
import re

parser = argparse.ArgumentParser(description='Extract relationships from text')
parser.add_argument('text', type=str, help='Text to extract relationships from')
args = parser.parse_args()

# Load the tokenizer
tokenizer = BertTokenizerFast.from_pretrained('model')

# Load the label dictionaries
with open('model/label_dicts.pkl', 'rb') as f:
    label_dicts = torch.load(f)

ner_label2idx, ner_idx2label = label_dicts['ner']
re_label2idx, re_idx2label = label_dicts['re']

# Define a regular expression pattern to match sentence boundaries
pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')

# Split the text into sentences
sentences = pattern.split(args.text)

# Initialize lists to store the extracted relationships
subject_texts = []
object_texts = []
re_labels = []

# Iterate over each sentence and extract the relationships
for sentence in sentences:
    encoding = tokenizer(sentence, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
    input_ids = encoding['input_ids'].squeeze()
    attention_mask = encoding['attention_mask'].squeeze()

    # Run the model on the sentence
    model = NER_RE_Model(len(ner_label2idx), len(re_label2idx), ner_label2idx, re_label2idx)
    model.load_state_dict(torch.load('model/custom_head.pt'))
    model.bert.requires_grad = False
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))

    # Extract the predicted subjects and objects
    ner_logits = outputs['ner_logits'].squeeze(0)
    subject_preds = torch.argmax(ner_logits[:, 1:len(ner_label2idx)-1], dim=-1) + 1
    object_preds = torch.argmax(ner_logits[:, len(ner_label2idx)+1:len(ner_label2idx)+len(ner_label2idx)-1], dim=-1) + len(ner_label2idx) + 1
    subject_positions = torch.where(subject_preds != 0)[0]
    object_positions = torch.where(object_preds != 0)[0]

    # Extract the predicted relationships
    for subject_position in subject_positions:
        subject_text = tokenizer.decode(input_ids[subject_position:object_positions[-1]+1], skip_special_tokens=True)
        for object_position in object_positions:
            if object_position > subject_position:
                object_text = tokenizer.decode(input_ids[object_position:subject_position+len(subject_text)], skip_special_tokens=True)
                input_ids_re, attention_mask_re = model.generate_inputs(subject_text, object_text)
                with torch.no_grad():
                    re_logits = model(input_ids=input_ids_re.unsqueeze(0), attention_mask=attention_mask_re.unsqueeze(0))['re_logits']
                re_pred = torch.argmax(re_logits.squeeze(0))
                if re_pred != 0:
                    subject_texts.append(subject_text)
                    object_texts.append(object_text)
                    re_labels.append(re_idx2label[re_pred])

# Print the extracted relationships
for i in range(len(subject_texts)):
    print(f"{subject_texts[i]} - {re_labels[i]} - {object_texts[i]}")
