import torch
from torch import nn
from transformers import BertModel

class NER_RE_Model(nn.Module):
    def __init__(self, ner_label_count, re_label_count):
        super(NER_RE_Model, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.ner_classifier = nn.Linear(self.bert.config.hidden_size, ner_label_count)
        self.re_classifier = nn.Linear(self.bert.config.hidden_size, re_label_count)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        ner_logits = self.ner_classifier(pooled_output)
        re_logits = self.re_classifier(pooled_output)
        return ner_logits, re_logits
