# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn.utils import rnn

from ...modules.common import LinearSequential


class NetworkV1(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_classes):
        super(NetworkV1, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.model = LinearSequential(
            in_features=2 * hidden_size,
            hidden_units=[512, 1024, 1024, 128],
            out_features=num_classes
        )

    def forward(self, x):
        """
        前向过程
        :param x: LongTensor 或者 (LongTensor,FloatTensor)
            LongTensor [N,T] N个文本，每个文本T个token
            FloatTensor [N,T] mask矩阵，实际值位置为1，填充值为0
        :return:
        """
        mask = None
        if isinstance(x, tuple) or isinstance(x, list):
            if len(x) >= 2:
                mask = x[1]
            x = x[0]

        # 1. embedding --> 静态词向量的方式 --> 每个token只会存在一个对应的词向量，不会考虑上下文
        x = self.embedding(x)  # [N,T] -> [N,T,hidden_size]

        # 2. 提取token的动态embedding --> 动态词向量 --> 每个token最终产生一个词向量，但是这个词向量在不同的上下文中，向量值可能是不一样
        # lstm_output: [N,T,2*hidden_size] 对应每个token的输出向量
        # lstm_state：lstm的状态信息
        lstm_output, lstm_state = self.lstm(x)
        x = lstm_output

        # 3. 合并T个token的向量，从而得到文本特征向量
        if mask is None:
            x = torch.mean(x, dim=1)  # [N,T,hidden_size] ->  [N,hidden_size]
        else:
            mask = mask.to(dtype=x.dtype, device=x.device)
            x = x * mask[..., None]  # [N,T,hidden_size]
            x = torch.sum(x, dim=1) / torch.sum(mask, dim=1, keepdim=True)  # [N,hidden_size]

        # 4. 全连接得到每个样本属于各个类别的置信度
        return self.model(x)


class NetworkV2(NetworkV1):
    def forward(self, x):
        mask = None
        if isinstance(x, tuple) or isinstance(x, list):
            if len(x) >= 2:
                mask = x[1]
            x = x[0]

        # 1. embedding --> 静态词向量的方式 --> 每个token只会存在一个对应的词向量，不会考虑上下文
        x = self.embedding(x)  # [N,T] -> [N,T,hidden_size]

        # 2. 提取token的动态embedding --> 动态词向量 --> 每个token最终产生一个词向量，但是这个词向量在不同的上下文中，向量值可能是不一样
        # lstm_output: [N,T,2*hidden_size] 对应每个token的输出向量
        # lstm_state：lstm的状态信息
        if mask is not None:
            x = rnn.pack_padded_sequence(
                x,  # [N,T,E]
                mask.sum(dim=1),  # [N] 每个样本的token实际长度
                batch_first=True,
                enforce_sorted=False  # 给定样本的长度是否是降序的
            )
        lstm_output, lstm_state = self.lstm(x)
        if mask is not None:
            lstm_output, _ = rnn.pad_packed_sequence(lstm_output, batch_first=True)
        x = lstm_output

        # 3. 合并T个token的向量，从而得到文本特征向量
        if mask is None:
            x = torch.mean(x, dim=1)  # [N,T,hidden_size] ->  [N,hidden_size]
        else:
            mask = mask.to(dtype=x.dtype, device=x.device)
            x = x * mask[..., None]  # [N,T,hidden_size]
            x = torch.sum(x, dim=1) / torch.sum(mask, dim=1, keepdim=True)  # [N,hidden_size]

        # 4. 全连接得到每个样本属于各个类别的置信度
        return self.model(x)
