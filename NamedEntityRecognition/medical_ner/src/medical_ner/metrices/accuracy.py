# -*- coding: utf-8 -*-
import torch


def acc(y_pred, y_true):
    y_pred = y_pred[0]
    y_pred = torch.argmax(y_pred, dim=-1)
    acc = torch.sum(y_pred.eq(y_true)).item() / y_true.numel()
    return {'acc': acc}


def build_acc2(model):
    def acc2(y_pred, y_true):
        emission_score, attention_mask = y_pred
        # noinspection PyUnresolvedReferences
        y_pred = model.decode(emission_score, attention_mask).to(y_true.dtype)  # 预测结果/标签的提取
        acc = torch.sum(y_pred.eq(y_true)).item() / y_true.numel()
        return {'acc2': acc}

    return acc2


def build_acc3(model):
    def acc3(y_pred, y_true):
        emission_score, attention_mask = y_pred
        # noinspection PyUnresolvedReferences
        y_pred = model.decode(emission_score, attention_mask).to(y_true.dtype)  # 预测结果/标签的提取

        y_true_mask = y_true.gt(0).float()
        total_y_true_tag = y_true_mask.sum().item()
        if total_y_true_tag > 0:
            acc = torch.sum(y_pred.eq(y_true).float() * y_true_mask).item() / total_y_true_tag
        else:
            acc = -1.0
        return {'acc3': acc}

    return acc3
