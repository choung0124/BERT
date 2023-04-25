import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the fine-tuned BERT RE model
re_model_dir = "models/re"
tokenizer = BertTokenizer.from_pretrained(re_model_dir)
re_model = BertForSequenceClassification.from_pretrained(re_model_dir)

# Set the threshold for RE predictions
threshold = 0.5

def extract_relationships(input_text, entities):
    # Generate all possible subject-object pairs
    pairs = []
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            if entities[i][1][2:] != entities[j][1][2:]:
                pairs.append((entities[i], entities[j]))

    # Tokenize the pairs and predict RE labels
    re_input_ids = []
    re_attention_masks = []

    for pair in pairs:
        subject = pair[0][0]
        obj = pair[1][0]
        tokens = tokenizer.tokenize(f"{subject} [SEP] {obj}")
        encoded = tokenizer.encode_plus(tokens, add_special_tokens=True, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        re_input_ids.append(encoded["input_ids"])
        re_attention_masks.append(encoded["attention_mask"])

    re_input_ids = torch.cat(re_input_ids, dim=0)
    re_attention_masks = torch.cat(re_attention_masks, dim=0)

    # Predict RE labels for the pairs
    re_outputs = re_model(re_input_ids, attention_mask=re_attention_masks)
    predicted_probs = torch.softmax(re_outputs[0], dim=1)[:, 1].tolist()

    # Extract related subject-object pairs based on the predicted RE labels and threshold
    related_pairs = []
    for i in range(len(pairs)):
        if predicted_probs[i] > threshold:
            related_pairs.append(pairs[i])

    # Generate the relationships for the related subject-object pairs
    relationships = []
    for pair in related_pairs:
        subject_type = pair[0][1][2:]
        obj_type = pair[1][1][2:]
        relation = re_model.config.id2label[1]
        relationships.append((pair[0][0], relation, pair[1][0]))

    return relationships
