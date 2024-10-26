# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch4keras.model import BaseModel
from transformers import T5EncoderModel, T5Config

from bert4torch.layers import CRF
from bert4torch.models import build_transformer_model


class BertCrfNerModel(BaseModel):
    def __init__(self, config_path, checkpoint_path, num_tags, model='bert', bert=None):
        super().__init__()
        if bert is None:
            bert = build_transformer_model(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                segment_vocab_size=0,
                model=model
            )
        self.bert = bert
        self.fc = nn.Linear(self.bert.config.hidden_size, num_tags)  # 包含首尾
        self.crf = CRF(num_tags)  # 在softmax的基础上额外增加一个类别到类别的转换概率矩阵

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

    def decode(self, emission_score, attention_mask):
        return self.crf.decode(emission_score, attention_mask)  # [btz, seq_len]
        # softmax loss 也就是预测的时候选择每个时刻预测属于各个类别置信度最大的作为预测结果，但是这个过程中并没有考虑整个序列的特性
        # eg: B-解剖部位  这个tag的后面 只能够是 I-解剖部位 或者 其它独立token/开始token，但是不能是 I-手术、I-疾病和诊断...
        # return torch.argmax(emission_score, dim=-1)

    def build_loss_fn(self):
        crf = self.crf

        class Loss(nn.Module):
            # noinspection PyMethodMayBeStatic
            def forward(self, outputs, labels):
                return crf(*outputs, labels)  # 计算CRF的损失

        return Loss()


class AlBertCrfNerModel(BertCrfNerModel):
    def __init__(self, config_path, checkpoint_path, num_tags):
        super(AlBertCrfNerModel, self).__init__(config_path, checkpoint_path, num_tags, model='albert')


class T5CrfNerModel(BertCrfNerModel):
    def __init__(self, config_path, checkpoint_path, num_tags):
        super(T5CrfNerModel, self).__init__(config_path, checkpoint_path, num_tags, model='t5_encoder')

    def forward(self, token_ids):
        sequence_output = self.bert([token_ids])  # [btz, seq_len, hdsz]
        emission_score = self.fc(sequence_output[0])  # [btz, seq_len, tag_size]
        attention_mask = token_ids.gt(0).long()  # 构建mask矩阵
        return emission_score, attention_mask


class T5CrfNerModel2(BertCrfNerModel):
    def __init__(self, config_path, checkpoint_path, num_tags):
        # bert = T5EncoderModel.from_pretrained(checkpoint_path)
        bert = T5EncoderModel(T5Config.from_pretrained(checkpoint_path))
        super().__init__(config_path, checkpoint_path, num_tags, bert=bert)

    def forward(self, token_ids):
        sequence_output = self.bert(token_ids)  # [btz, seq_len, hdsz]
        emission_score = self.fc(sequence_output[0])  # [btz, seq_len, tag_size]
        attention_mask = token_ids.gt(0).long()  # 构建mask矩阵
        return emission_score, attention_mask
