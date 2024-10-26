# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.optim as optim


def get_train_optim(net: nn.Module, lr, name='sgd', **kwargs):
    gw, gb = [], []
    for param_name, param in net.named_parameters():
        if param.requires_grad is False:
            continue
        if 'bias' in param_name:
            gb.append(param)
        else:
            gw.append(param)

    if name == 'sgd':
        train_op = optim.SGD(
            params=gb, lr=lr,
            momentum=kwargs.get('momentum', 0),  # 动量法的累计梯度的系数
            dampening=kwargs.get('dampening', 0),  # 动量法中当前梯度的系数值(1-dampening)
            nesterov=kwargs.get('nesterov', False),  # 牛顿动量法
            weight_decay=0.0  # 针对bias不进行惩罚性限制
        )
        train_op.add_param_group(
            param_group={
                "params": gw,
                "lr": train_op.defaults['lr'] * 0.1,
                "weight_decay": kwargs.get('weight_decay', 0.0)
            }
        )
    elif name == 'adam':
        train_op = optim.Adam(params=gb, lr=lr)
        train_op.add_param_group(
            param_group={
                "params": gw, "lr": train_op.defaults['lr'] * 0.1,
                "weight_decay": kwargs.get('weight_decay', 0.0)
            }
        )
    elif name == 'adamw':
        train_op = optim.AdamW(params=gb, lr=lr, weight_decay=0)
        train_op.add_param_group(
            param_group={
                "params": gw, "lr": train_op.defaults['lr'] * 0.1,
                "weight_decay": kwargs.get('weight_decay', 0.002)
            }
        )
    else:
        raise ValueError(f"当前不支持该优化器:{name}")
    return train_op


def get_optim_lr_scheduler(opt, name="linear"):
    if name is None:
        return None
    elif isinstance(name, str):
        name = name.lower()
        if name == 'linear':
            scheduler = optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.1, total_iters=10)
        else:
            raise ValueError(f"当前不支持该学习率变化方式:{name}")
    else:
        # noinspection PyTypeChecker
        scheduler = optim.lr_scheduler.LambdaLR(opt, name)
    return scheduler
