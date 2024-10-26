# -*- coding: utf-8 -*-
import torch.nn as nn


def get_loss_fn(loss="cross_entropy_loss", weight=None, label_smoothing=None):
    if loss == 'cross_entropy_loss':
        loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    else:
        raise ValueError(f"当前损失不支持:{loss}")
    return loss_fn
