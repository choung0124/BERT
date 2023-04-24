import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from model_definition import NER_RE_Model
from data_prep import tokenize_data, create_data_loader, NERRE_Dataset, ner_label2idx, re_label2idx, filtered_data, idx2ner_label, idx2re_label

# Create the eval_dataset with filtered data
eval_dataset = NERRE_Dataset(*zip(*filtered_data))

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
tokenized_eval_data = tokenize_data(eval_dataset)
eval_data_loader = DataLoader(tokenized_eval_data, batch_size=8, shuffle=False)

model = NER_RE_Model(len(ner_label2idx), len(re_label2idx))

model_dir = "model"
model.bert.from_pretrained(model_dir)
custom_head = torch.load(os.path.join(model_dir, 'custom_head.pt'))
model.ner_classifier.load_state_dict(custom_head['ner_classifier'])
model.re_classifier.load_state_dict(custom_head['re_classifier'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

def evaluate(model, data_loader, device):
    ner_correct_predictions = 0
    re_correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, ner_label, re_label = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            ner_labels = ner_label.to(device)
            re_labels = re_label.to(device)

            ner_logits, re_logits = model(input_ids, attention_mask)
            ner_predictions = torch.argmax(ner_logits, dim=1)
            re_predictions = torch.argmax(re_logits, dim=1)

            ner_correct_predictions += (ner_predictions == ner_labels).sum().item()
            re_correct_predictions += (re_predictions == re_labels).sum().item()
            total_samples += ner_labels.size(0)

    ner_accuracy = ner_correct_predictions / total_samples
    re_accuracy = re_correct_predictions / total_samples
    return ner_accuracy, re_accuracy

ner_accuracy, re_accuracy = evaluate(model, eval_data_loader, device)
print(f"NER accuracy: {ner_accuracy:.4f}")
print(f"RE accuracy: {re_accuracy:.4f}")
