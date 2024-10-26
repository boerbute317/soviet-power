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
from text_classify.task.task3.dataset import fetch_dataloader
from text_classify.task.task3.network import Network
from text_classify.trainer.base import Trainer

if __name__ == '__main__':
    # 初始化
    logging.basicConfig(level=logging.INFO)
    FILE_ROOT_DIR = os.path.dirname(__file__)
    now_str = datetime.now().strftime("%Y%m%d%H%M%S")
    train_dataloader, test_dataloader, token_2_idx, num_classes, class_weight, strokengram2id = fetch_dataloader(
        os.path.join(FILE_ROOT_DIR, "../datas"),
        16
    )
    net = Network(
        vocab_size=len(token_2_idx),
        hidden_size=128,
        num_classes=num_classes,
        stroke_size=len(strokengram2id)  # 笔画n-gram的总数
    )
    # NOTE: 能不能每个epoch执行完后，根据评估数据集的效果，来调整各个类别的损失权重值
    loss_fn = get_loss_fn(
        weight=torch.tensor(class_weight, dtype=torch.float32).view(-1),
        label_smoothing=0.1
    )
    train_opt = get_train_optim(
        net, 0.001,
        name='adamw', momentum=0.1, nesterov=True,
        weight_decay=0.002
    )

    # lr_scheduler = get_optim_lr_scheduler(train_opt, name="linear")
    # lr_scheduler = get_optim_lr_scheduler(train_opt, name=lambda epoch: 0.9 ** epoch)
    # lr_scheduler = get_optim_lr_scheduler(train_opt, name=_lr_update_func)

    trainer = Trainer(
        net, loss_fn, train_opt,
        train_dataloader, test_dataloader,
        total_epoch=100,
        summary_log_dir=f'{FILE_ROOT_DIR}/output/{now_str}/summary',
        save_dir=f'{FILE_ROOT_DIR}/output/{now_str}/model',
        example_inputs=torch.randint(100, size=(2, 5))
    )

    # 训练
    trainer.fit()
