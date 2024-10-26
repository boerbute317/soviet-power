# -*- coding: utf-8 -*-
import logging
import os
import sys
from datetime import datetime

import torch

print(sys.path)

# 这里可以直接以下列代码import Trainer的主要原因是：sys.path环境变量的文件夹中有text_classify
from text_classify.loss import get_loss_fn
from text_classify.optim import get_train_optim, get_optim_lr_scheduler
from text_classify.task.task0.dataset import fetch_dataloader
from text_classify.task.task0.network import Network
from text_classify.trainer.base import Trainer


def _lr_update_func(_epoch):
    """
    自定义学习率变化函数，该函数入参为epoch的值: 调用学习率step方法的次数；返回的是当前epoch对应的学习率系数
    最终学习率为：base_lr * 学习率系数
    :param _epoch:
    :return:
    """
    if _epoch < 5:
        return 0.5 + (1.0 - 0.5) * _epoch / 5
    elif _epoch < 8:
        return 1.0
    else:
        return 0.8 ** (_epoch - 8)


if __name__ == '__main__':
    # 初始化
    logging.basicConfig(level=logging.INFO)
    FILE_ROOT_DIR = os.path.dirname(__file__)
    now_str = datetime.now().strftime("%Y%m%d%H%M%S")
    train_dataloader, test_dataloader, num_features, num_classes, class_weight = fetch_dataloader(
        os.path.join(FILE_ROOT_DIR, "../datas"),
        16
    )
    net = Network(num_features, num_classes)
    # NOTE: 能不能每个epoch执行完后，根据评估数据集的效果，来调整各个类别的损失权重值
    loss_fn = get_loss_fn(
        weight=torch.tensor(class_weight, dtype=torch.float32).view(-1),
        label_smoothing=0.1
    )
    train_opt = get_train_optim(
        net, 0.1,
        name='sgd', momentum=0.1, nesterov=True,
        weight_decay=0.002
    )

    # lr_scheduler = get_optim_lr_scheduler(train_opt, name="linear")
    # lr_scheduler = get_optim_lr_scheduler(train_opt, name=lambda epoch: 0.9 ** epoch)
    lr_scheduler = get_optim_lr_scheduler(train_opt, name=_lr_update_func)

    trainer = Trainer(
        net, loss_fn, train_opt,
        train_dataloader, test_dataloader,
        total_epoch=100,
        summary_log_dir=f'{FILE_ROOT_DIR}/output/{now_str}/summary',
        save_dir=f'{FILE_ROOT_DIR}/output/{now_str}/model',
        example_inputs=torch.randn(1, num_features),
        lr_scheduler=lr_scheduler
    )

    # 训练
    trainer.fit()
