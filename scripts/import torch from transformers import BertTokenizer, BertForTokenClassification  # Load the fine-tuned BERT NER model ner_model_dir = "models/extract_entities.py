import torch
from transformers import BertTokenizer, BertForTokenClassification

# Load the fine-tuned BERT NER model
ner_model_dir = "models/ner"
tokenizer = BertTokenizer.from_pretrained(ner_model_dir)
ner_model = BertForTokenClassification.from_pretrained(ner_model_dir)

def extract_entities(input_text):
    # Tokenize the input text and predict NER labels
    input_ids = torch.tensor(tokenizer.encode(input_text, add_special_tokens=True)).unsqueeze(0)
    ner_outputs = ner_model(input_ids)[0]
    predicted_labels = torch.argmax(ner_outputs, dim=2).squeeze(0).tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())

    # Extract entities from the predicted NER labels
    entities = []
    current_entity = None
    current_entity_type = None

    for i in range(len(tokens)):
        if predicted_labels[i] == 1:
            if current_entity is not None:
                entities.append((current_entity, current_entity_type))
            current_entity = tokens[i]
            current_entity_type = "B-" + ner_model.config.id2label[predicted_labels[i]]
        elif predicted_labels[i] == 2:
            current_entity += " " + tokens[i]
        else:
            if current_entity is not None:
                entities.append((current_entity, current_entity_type))
                current_entity = None
                current_entity_type = None

    if current_entity is not None:
        entities.append((current_entity, current_entity_type))

    return entities
