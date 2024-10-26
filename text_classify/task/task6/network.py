# -*- coding: utf-8 -*-
import torch.nn as nn
from transformers import BertModel

from ...modules.common import LinearSequential


class NetworkBertV1(nn.Module):
    def __init__(self, bert_model_id, num_classes, bert_freeze=True):
        super(NetworkBertV1, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_id)
        if bert_freeze:
            print("将bert相关参数进行冻结(冻结后，反向传播不会更新冻结参数)")
            for name, param in self.bert.named_parameters():
                print(f"冻结参数:{name}")
                param.requires_grad = False
        self.model = LinearSequential(
            in_features=self.bert.config.hidden_size,
            hidden_units=[512, 128],
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

        # 1. 动态词向量的提取
        # [N,T] -> [N,T,hidden_size]
        bert_output = self.bert(
            input_ids=x,
            attention_mask=mask,
            output_hidden_states=True,  # 每一层的输出是否均返回 hidden_states
            return_dict=False  # 返回对象的类型是否是字典对象
        )

        # # 获取 last_hidden_state 对应的特征向量
        # bert_output = bert_output[0]  # [N,T,hidden_size]
        # x = bert_output[:, 0, :]  # 提取[CLS]对应的token特征向量
        # 直接获取pooler_output对应的特征向量
        x = bert_output[1]  # [N,hidden_size]
        if x is None:
            bert_output = bert_output[0]  # [N,T,hidden_size]
            x = bert_output[:, 0, :]

        # 4. 全连接得到每个样本属于各个类别的置信度
        return self.model(x)
