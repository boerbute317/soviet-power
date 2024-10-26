# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch4keras.model import BaseModel

from bert4torch.models import build_transformer_model


class BertSoftmaxNerModel(BaseModel):
    def __init__(self, config_path, checkpoint_path, num_tags):
        super().__init__()
        self.bert = build_transformer_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            segment_vocab_size=0,
            model='bert'
        )
        self.fc = nn.Linear(768, num_tags)  # 包含首尾
        self.num_tags = num_tags

    def forward(self, token_ids):
        sequence_output = self.bert([token_ids])  # [btz, seq_len, hdsz]
        emission_score = self.fc(sequence_output)  # [btz, seq_len, tag_size]
        attention_mask = token_ids.gt(0).long()  # 构建mask矩阵
        return emission_score, attention_mask

    def predict(self, token_ids):
        self.eval()
        with torch.no_grad():
            emission_score, attention_mask = self.forward(token_ids)
            best_path = self.decode(emission_score, attention_mask)
        return best_path

    # noinspection PyMethodMayBeStatic
    def decode(self, emission_score, attention_mask):
        best_path = torch.argmax(emission_score, dim=-1)  # [btz, seq_len]
        best_path = best_path * attention_mask.to(dtype=best_path.dtype)
        return best_path

    def build_loss_fn(self):
        num_tags = self.num_tags
        loss_fn = nn.CrossEntropyLoss()

        class Loss(nn.Module):
            # noinspection PyMethodMayBeStatic
            def forward(self, outputs, labels):
                emission_score, attention_mask = outputs
                return loss_fn(emission_score.view(-1, num_tags), labels.view(-1))

        return Loss()
