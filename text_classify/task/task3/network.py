# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from ...modules.common import LinearSequential


# noinspection DuplicatedCode
class Network(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_classes, stroke_size):
        super(Network, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.stroke_embedding = nn.Embedding(num_embeddings=stroke_size, embedding_dim=hidden_size)
        self.stroke_merge_linear = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU()
        )
        self.model = LinearSequential(
            in_features=hidden_size,
            hidden_units=[512, 1024, 1024, 128],
            out_features=num_classes
        )

    def forward(self, entity):
        """
        前向过程
        :param entity: LongTensor 或者 (LongTensor,FloatTensor) 或者 (LongTensor,FloatTensor,LongTensor,FloatTensor)
            第一个元素 LongTensor [N,T] N个文本，每个文本T个token
            第二个元素 FloatTensor [N,T] mask矩阵，实际值位置为1，填充值为0
            第三个元素 LongTensor [N,T,L] N个文本，每个文本T个token，每个token对应L个笔画n_gram
            第四个元素 FloatTensor [N,T,L] mask矩阵，笔画n_gram实际存在对应值为1，填充为0
        :return:
        """
        token = entity
        token_mask = None
        stroke = None
        stroke_mask = None
        if isinstance(entity, tuple) or isinstance(entity, list):
            if len(entity) >= 2:
                token = entity[0]
                token_mask = entity[1]
            if len(entity) >= 4:
                stroke = entity[2]
                stroke_mask = entity[3]

        # token对应的文本特征向量
        sentence_x = self.get_token_sentence_embedding(token, token_mask)  # [N,E]
        if stroke is not None:
            # 笔画对应的文本特征向量
            stroke_sentence_x = self.get_stroke_sentence_embedding(stroke, stroke_mask)  # [N,E]
            # 合并token向量和笔画向量
            sentence_x = torch.concat([sentence_x, stroke_sentence_x], dim=1)  # [N,2E]
            sentence_x = self.stroke_merge_linear(sentence_x)

        # 3. 全连接得到每个样本属于各个类别的置信度
        return self.model(sentence_x)

    def get_stroke_sentence_embedding(self, stroke, stroke_mask):
        # 1. embedding操作
        x = self.stroke_embedding(stroke)  # [N,T,L] -> [N,T,L,hidden_size]
        # 2. 合并T个token的向量，从而得到文本特征向量
        if stroke_mask is None:
            x = x.mean(-2).mean(-2)  # [N,T,L,hidden_size] -> [N,T,hidden_size] -> [N,hidden_size]
        else:
            mask = stroke_mask.to(dtype=x.dtype, device=x.device)
            mask = mask[..., None]  # [N,T,L,1]
            x = x * mask  # [N,T,L,hidden_size]
            x = x.sum(-2).sum(-2) / mask.sum(-2).sum(-2)  # [N,hidden_size] / [N,1] ==> [N,hidden_size]
        return x

    def get_token_sentence_embedding(self, token, token_mask):
        # 1. embedding
        x = self.embedding(token)  # [N,T] -> [N,T,hidden_size]
        # 2. 合并T个token的向量，从而得到文本特征向量
        if token_mask is None:
            x = torch.mean(x, dim=1)  # [N,T,hidden_size] ->  [N,hidden_size]
        else:
            mask = token_mask.to(dtype=x.dtype, device=x.device)
            x = x * mask[..., None]  # [N,T,hidden_size]
            x = torch.sum(x, dim=1) / torch.sum(mask, dim=1, keepdim=True)  # [N,hidden_size]
        return x
