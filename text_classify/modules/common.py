# -*- coding: utf-8 -*-
import torch.nn as nn


def get_activate_function(act):
    if act is None:
        return nn.Identity()
    elif isinstance(act, str):
        act = act.lower()
        if act == 'relu':
            return nn.ReLU()
        else:
            raise ValueError(f"当前不支持该激活函数:{act}")
    else:
        return act


class LinearModule(nn.Module):
    def __init__(self, in_features, out_features, act='relu'):
        super(LinearModule, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.act = get_activate_function(act)

    def forward(self, x):
        return self.act(self.linear(x))


class LinearSequential(nn.Module):
    def __init__(self, in_features, hidden_units, out_features, hidden_act='relu', out_act=None):
        super(LinearSequential, self).__init__()
        layers = []
        for unit in hidden_units:
            layers.append(LinearModule(in_features=in_features, out_features=unit, act=hidden_act))
            in_features = unit
        layers.append(LinearModule(in_features=in_features, out_features=out_features, act=out_act))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
