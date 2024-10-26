# -*- coding: utf-8 -*-

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

FILE_ROOT_DIR = os.path.dirname(__file__)
now_str = datetime.now().strftime("%Y%m%d%H%M%S")

# %% 数据预处理: x的分词、y的标签转换 --> 直接和04文件共享

# %% 参与训练的处理后的数据加载

df = pd.read_csv(os.path.join(FILE_ROOT_DIR, "../datas/split_train.csv"), sep="\t", header=None, names=['x', 'y'])
# df = df[(df.y == 1) | (df.y == 2)]
X_train, X_test, y_train, y_test = train_test_split(df.x.values, df.y.values, test_size=0.2, random_state=28)
with open(os.path.join(FILE_ROOT_DIR, "../datas/label2idx.json"), "r", encoding="utf-8") as reader:
    ylabel2idx = json.load(reader)
    yidx2label = {v: k for k, v in ylabel2idx.items()}

# %% 文本向量转换

vectorizer = TfidfVectorizer(
    sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english"
)
# 输入是list[str] 输出是特征属性矩阵,shape为[N,E]
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

# %% 模型构建、损失函数构建、优化器构建

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

# %% 模型的训练优化、评估、可视化

total_epoch = 100  # epoch指的是从头到尾遍历一次叫一个epoch，一个epoch包含多个batch
batch_size = 16
total_train_samples = len(X_train_vec)
total_batch = total_train_samples // batch_size

# pip install tensorboard==2.12.3
# 在命令行执行tensorboard命令:
# tensorboard --logdir D:\workspaces\study\NLP02\text_classify\demo\output --host 127.0.0.1 --port 6006
writer = SummaryWriter(log_dir=f'{FILE_ROOT_DIR}/output/{now_str}/summary')

train_step = 0
for epoch in range(total_epoch):
    print(epoch, "=" * 50)
    # 训练
    net.train()
    random_indecs = np.random.permutation(total_train_samples)
    for batch_idx in range(total_batch):
        si = batch_idx * batch_size
        ei = si + batch_size
        index = random_indecs[si:ei]
        x = torch.from_numpy(X_train_vec[index]).to(torch.float32)
        y = torch.from_numpy(y_train[index]).to(torch.long)

        train_step += 1
        scores = net(x)  # 获取模型前向预测结果 [N,12]
        loss = loss_fn(scores, y)  # 求解损失
        train_op.zero_grad()  # 将每个参数的梯度重置为0
        loss.backward()  # 求解每个参数的梯度值
        train_op.step()  # 参数更新
        writer.add_scalar('train_loss', loss.item(), global_step=train_step)

        if batch_idx % 100 == 0:
            print(f"epoch {epoch}/{total_epoch} batch {batch_idx}/{total_batch} loss:{loss.item():.3f}")

    writer.add_embedding(net.layer[0].weight, global_step=epoch)
    # add_histogram可能存在异常:  {TypeError}No loop matching the specified signature and casting was found for ufunc greater
    # 异常解决方案为：https://blog.csdn.net/Williamcsj/article/details/135334545  修改源码
    writer.add_histogram('layer0.w', net.layer[0].weight.detach().numpy(), global_step=epoch)

    # 评估
    net.eval()
    with torch.no_grad():
        x = torch.from_numpy(X_test_vec).to(torch.float32)
        y = torch.from_numpy(y_test).to(torch.long)
        scores = net(x)  # 获取模型前向预测结果 [N,12]
        y_pred = torch.argmax(scores, dim=1)

        confusion_matrix = metrics.confusion_matrix(y.numpy(), y_pred.numpy())
        print(confusion_matrix)
        report = metrics.classification_report(y.numpy(), y_pred.numpy(), zero_division=0.0)
        print(report)

# 关闭日志输出的对象
writer.close()
