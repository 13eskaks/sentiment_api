from abc import ABC

from transformers import BertModel, PreTrainedModel, BertConfig
import torch.nn as nn


class BertForSentimentRegression(PreTrainedModel, ABC):
    def __init__(self, config: BertConfig, dropout=0.3):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(config.hidden_size, 3)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        return self.regressor(cls_output)
