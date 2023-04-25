import torch
from torch import nn
from transformers import BertModel

class NER_RE_Model(nn.Module):
    def __init__(self, ner_label_count, re_label_count):
        super(NER_RE_Model, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.ner_classifier = nn.Linear(self.bert.config.hidden_size, ner_label_count)
        self.re_classifier = nn.Linear(self.bert.config.hidden_size * 2, re_label_count)

    def forward(self, input_ids, attention_mask, entity_positions):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        ner_logits = self.ner_classifier(sequence_output)

        entity_outputs = sequence_output[torch.arange(input_ids.shape[0]).unsqueeze(1), entity_positions]
        entity_pair_output = torch.cat((entity_outputs[:, 0], entity_outputs[:, 1]), dim=1)
        re_logits = self.re_classifier(entity_pair_output)

        return ner_logits, re_logits
