# -*- coding: utf-8 -*-
import copy
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class ListDataset(Dataset):
    def __init__(self, X, Y):
        super(ListDataset, self).__init__()
        self.X = X  # list(list(int)) 里面的存储的是每个文本对应的token id列表
        self.Y = Y  # list(int) 里面存储的是每个文本对应的类别id

    def __len__(self):
        """
        返回当前数据集的样本数目
        :return:
        """
        return len(self.X)

    def __getitem__(self, i):
        """
        基于给定的样本索引i获取对应的对象
        :param i: 样本index
        :return:
        """
        _x = self.X[i]  # token id 列表  --> list(int)  --> 不同样本，这个token id列表返回长度不一样
        _y = self.Y[i]  # 数字  --> int
        return copy.deepcopy(_x), _y, len(_x)

    @staticmethod
    def collate_fn(batch):
        batch_x, batch_y, batch_x_length = list(zip(*batch))
        max_length = max(batch_x_length)
        mask = []
        for i in range(len(batch_x)):
            _x = batch_x[i]
            _mask = np.zeros(max_length)
            _mask[:len(_x)] = 1
            mask.append(list(_mask))
            if len(_x) < max_length:
                _x.extend([0] * (max_length - len(_x)))

        batch_x = torch.tensor(batch_x, dtype=torch.int64)  # [N,T]
        batch_y = torch.tensor(batch_y, dtype=torch.int64)  # [N]
        mask = torch.tensor(mask, dtype=torch.float32)  # [N,T] 实际值位置为1，填充值位置为0
        return (batch_x, mask), batch_y


def fetch_dataloader(data_path_dir, batch_size):
    data_path = os.path.join(data_path_dir, "split_train.csv")
    df = pd.read_csv(data_path, sep="\t", header=None, names=['x', 'y'])

    # 构建词典 token和token id的映射mapping
    token_2_idx = {
        '<PAD>': 0, '<UNK>': 1
    }
    X = []
    for text in df.x.values:
        x = []
        for token in text.split(" "):
            try:
                token_id = token_2_idx[token]
            except KeyError:
                token_id = len(token_2_idx)
                token_2_idx[token] = token_id
            x.append(token_id)
        X.append(x)

    X_train, X_test, y_train, y_test = train_test_split(X, list(df.y.values), test_size=0.2, random_state=28)
    with open(os.path.join(data_path_dir, "label2idx.json"), "r", encoding="utf-8") as reader:
        ylabel2idx = json.load(reader)
        yidx2label = {v: k for k, v in ylabel2idx.items()}

    cnts = df.y.value_counts()[list(range(len(yidx2label)))]
    total_cnts = cnts.sum()
    weights = total_cnts / (cnts + 1)
    weights = weights.clip(len(cnts), 1.2 * len(cnts))
    weights = torch.softmax(torch.tensor(weights.values), 0)

    train_dataset = ListDataset(X_train, y_train)
    test_dataset = ListDataset(X_test, y_test)
    # DataLoader的功能是将Dataset对象中的一条一条的数据组成成一个批次的数据
    train_dataloader = DataLoader(
        dataset=train_dataset,  # 给定数据集对象
        batch_size=batch_size,  # 批次大小，也就是将多少个样本组成一个批次
        shuffle=True,  # 是否打乱数据加载的顺序，默认是不打乱
        collate_fn=ListDataset.collate_fn,  # 是一个方法，功能是定义了如何将dataset返回的对象转换/封装为批次对象
        drop_last=False,  # 如果最后一个批次的样本数量小于batch_size, true表示删除，false表示不删除
        num_workers=0,  # 给定数据加载底层使用多少个子进程加载(并发的加载数据)， 0表示不进行并发加载
        # prefetch_factor=2  # 并发加载数据的时候，预加载多少数据，这个参数的默认值在不同pytorch版本中不一样，当num_workers=0的时候，该参数必须为默认值；当num_workers>0的时候，建议该参数设定为:num_workers * batch_size
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size * 2, collate_fn=ListDataset.collate_fn)

    return train_dataloader, test_dataloader, token_2_idx, len(yidx2label), weights
