# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from ...modules.common import LinearSequential


class Network(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_classes):
        super(Network, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.model = LinearSequential(
            in_features=hidden_size,
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

        # 1. embedding
        x = self.embedding(x)  # [N,T] -> [N,T,hidden_size]
        # 2. 合并T个token的向量，从而得到文本特征向量
        if mask is None:
            x = torch.mean(x, dim=1)  # [N,T,hidden_size] ->  [N,hidden_size]
        else:
            mask = mask.to(dtype=x.dtype, device=x.device)
            x = x * mask[..., None]  # [N,T,hidden_size]
            x = torch.sum(x, dim=1) / torch.sum(mask, dim=1, keepdim=True) # [N,hidden_size]
        # 3. 全连接得到每个样本属于各个类别的置信度
        return self.model(x)
