import torch
from torch import nn
from transformers import BertModel

class NER_RE_Model(nn.Module):
    def __init__(self, ner_label_count, re_label_count, ner_label2idx, re_label2idx):
        super(NER_RE_Model, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.ner_classifier = nn.Linear(self.bert.config.hidden_size, ner_label_count)
        self.re_classifier = nn.Linear(self.bert.config.hidden_size, re_label_count)
        self.ner_label2idx = ner_label2idx
        self.re_label2idx = re_label2idx

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        ner_logits = self.ner_classifier(last_hidden_state)
        re_logits = self.re_classifier(outputs.pooler_output)
        return {'ner_logits': ner_logits, 're_logits': re_logits}


