# -*- coding: utf-8 -*-

import json
import os
from datetime import datetime

import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

FILE_ROOT_DIR = os.path.dirname(__file__)
now_str = datetime.now().strftime("%Y%m%d%H%M%S")

# %% 1. 数据预处理: x的分词、y的标签转换 --> 直接和04文件共享

pass

# %% 2. 参与训练的处理后的数据加载

df = pd.read_csv(os.path.join(FILE_ROOT_DIR, "../datas/split_train.csv"), sep="\t", header=None, names=['x', 'y'])
# df = df[(df.y == 1) | (df.y == 2)]
X_train, X_test, y_train, y_test = train_test_split(df.x.values, df.y.values, test_size=0.2, random_state=28)
with open(os.path.join(FILE_ROOT_DIR, "../datas/label2idx.json"), "r", encoding="utf-8") as reader:
    ylabel2idx = json.load(reader)
    yidx2label = {v: k for k, v in ylabel2idx.items()}

# %% 3.文本向量转换

vectorizer = TfidfVectorizer(
    sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english"
)
# 输入是list[str] 输出是特征属性矩阵,shape为[N,E]
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

# %% 4.模型构建、损失函数构建、优化器构建

import torch
import torch.nn as nn
import torch.optim as optim


class Network(nn.Module):
    def __init__(self, in_features, num_classes):
        super(Network, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.layer = nn.Sequential(
            nn.Linear(self.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes)
        )

    def forward(self, x):
        """
        前向过程
        :param x: [N, self.in_features] 做了基础向量转换后的输入
        :return: [N, self.num_classes] 每个样本对应各个类别的置信度
        """
        return self.layer(x)


net = Network(in_features=X_train_vec.shape[1], num_classes=len(yidx2label))
loss_fn = nn.CrossEntropyLoss()
train_op = optim.SGD(params=net.parameters(), lr=0.1)
# pip install tensorboard==2.12.3
# 在命令行执行tensorboard命令:
# tensorboard --logdir D:\workspaces\study\NLP02\text_classify\demo\output --host 127.0.0.1 --port 6006
writer = SummaryWriter(log_dir=f'{FILE_ROOT_DIR}/output/{now_str}/summary')


# %% 模型的训练优化、评估、可视化

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


def collate_fn(batch):
    """
    定义将一个批次的样本如何进行封装
    :param batch: list对象，list中的每个对象是Dataset的__getitem__方法返回的对象
    :return:
    """
    transposed = list(zip(*batch))
    return torch.stack(transposed[0], dim=0), torch.stack(transposed[1], dim=0)


batch_size = 16

train_dataset = NumpyDataset(X_train_vec, y_train)
test_dataset = NumpyDataset(X_test_vec, y_test)
# DataLoader的功能是将Dataset对象中的一条一条的数据组成成一个批次的数据
train_dataloader = DataLoader(
    dataset=train_dataset,  # 给定数据集对象
    batch_size=batch_size,  # 批次大小，也就是将多少个样本组成一个批次
    shuffle=True,  # 是否打乱数据加载的顺序，默认是不打乱
    collate_fn=collate_fn,  # 是一个方法，功能是定义了如何将dataset返回的对象转换/封装为批次对象
    drop_last=False,  # 如果最后一个批次的样本数量小于batch_size, true表示删除，false表示不删除
    num_workers=0,  # 给定数据加载底层使用多少个子进程加载(并发的加载数据)， 0表示不进行并发加载
    # prefetch_factor=2  # 并发加载数据的时候，预加载多少数据，这个参数的默认值在不同pytorch版本中不一样，当num_workers=0的时候，该参数必须为默认值；当num_workers>0的时候，建议该参数设定为:num_workers * batch_size
)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size * 2)

total_epoch = 100  # epoch指的是从头到尾遍历一次叫一个epoch，一个epoch包含多个batch

train_step = 0
for epoch in range(total_epoch):
    print(epoch, "=" * 50)
    # 训练
    net.train()
    for batch_x, batch_y in train_dataloader:
        train_step += 1
        scores = net(batch_x)  # 获取模型前向预测结果 [N,12]
        loss = loss_fn(scores, batch_y)  # 求解损失
        train_op.zero_grad()  # 将每个参数的梯度重置为0
        loss.backward()  # 求解每个参数的梯度值
        train_op.step()  # 参数更新
        writer.add_scalar('train_loss', loss.item(), global_step=train_step)

        if train_step % 100 == 0:
            print(f"epoch {epoch}/{total_epoch} batch {train_step} loss:{loss.item():.3f}")

    writer.add_embedding(net.layer[0].weight, global_step=epoch)
    # add_histogram可能存在异常:  {TypeError}No loop matching the specified signature and casting was found for ufunc greater
    # 异常解决方案为：https://blog.csdn.net/Williamcsj/article/details/135334545  修改源码
    writer.add_histogram('layer0.w', net.layer[0].weight.detach().numpy(), global_step=epoch)

    # 评估
    net.eval()
    with torch.no_grad():
        test_y_true, test_y_pred = [], []
        for batch_x, batch_y in test_dataloader:
            scores = net(batch_x)  # 获取模型前向预测结果 [N,12]
            y_pred = torch.argmax(scores, dim=1)
            test_y_true.append(batch_y)
            test_y_pred.append(y_pred)

        test_y_true = torch.concatenate(test_y_true, dim=0).numpy()
        test_y_pred = torch.concatenate(test_y_pred, dim=0).numpy()

        confusion_matrix = metrics.confusion_matrix(test_y_true, test_y_pred)
        print(confusion_matrix)
        report = metrics.classification_report(test_y_true, test_y_pred, zero_division=0.0)
        print(report)

# 关闭日志输出的对象
writer.close()
