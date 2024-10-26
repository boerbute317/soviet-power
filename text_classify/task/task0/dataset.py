# -*- coding: utf-8 -*-
import json
import os

import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class NumpyDataset(Dataset):
    def __init__(self, X, Y):
        super(NumpyDataset, self).__init__()
        self.X = X
        self.Y = Y

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
        _x = self.X[i]  # 一维向量
        _y = self.Y[i]  # 数字
        return torch.tensor(_x, dtype=torch.float32), torch.tensor(_y, dtype=torch.int64)


def fetch_dataloader(data_path_dir, batch_size):
    data_path = os.path.join(data_path_dir, "split_train.csv")
    df = pd.read_csv(data_path, sep="\t", header=None, names=['x', 'y'])
    X_train, X_test, y_train, y_test = train_test_split(df.x.values, df.y.values, test_size=0.2, random_state=28)
    with open(os.path.join(data_path_dir, "label2idx.json"), "r", encoding="utf-8") as reader:
        ylabel2idx = json.load(reader)
        yidx2label = {v: k for k, v in ylabel2idx.items()}

    cnts = df.y.value_counts()[list(range(len(yidx2label)))]
    total_cnts = cnts.sum()
    weights = total_cnts / (cnts + 1)
    weights = weights.clip(len(cnts), 1.2 * len(cnts))
    weights = torch.softmax(torch.tensor(weights.values), 0)

    vectorizer = TfidfVectorizer(
        sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english"
    )
    # 输入是list[str] 输出是特征属性矩阵,shape为[N,E]
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()

    train_dataset = NumpyDataset(X_train_vec, y_train)
    test_dataset = NumpyDataset(X_test_vec, y_test)
    # DataLoader的功能是将Dataset对象中的一条一条的数据组成成一个批次的数据
    train_dataloader = DataLoader(
        dataset=train_dataset,  # 给定数据集对象
        batch_size=batch_size,  # 批次大小，也就是将多少个样本组成一个批次
        shuffle=True,  # 是否打乱数据加载的顺序，默认是不打乱
        collate_fn=None,  # 是一个方法，功能是定义了如何将dataset返回的对象转换/封装为批次对象
        drop_last=False,  # 如果最后一个批次的样本数量小于batch_size, true表示删除，false表示不删除
        num_workers=0,  # 给定数据加载底层使用多少个子进程加载(并发的加载数据)， 0表示不进行并发加载
        # prefetch_factor=2  # 并发加载数据的时候，预加载多少数据，这个参数的默认值在不同pytorch版本中不一样，当num_workers=0的时候，该参数必须为默认值；当num_workers>0的时候，建议该参数设定为:num_workers * batch_size
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size * 2)

    return train_dataloader, test_dataloader, X_train_vec.shape[1], len(yidx2label), weights
