# -*- coding: utf-8 -*-


import torch.nn as nn

from ...modules.common import LinearSequential


class Network(nn.Module):
    def __init__(self, in_features, num_classes):
        super(Network, self).__init__()
        self.model = LinearSequential(
            in_features=in_features,
            hidden_units=[512, 1024, 1024, 128],
            out_features=num_classes
        )

    def forward(self, x):
        """
        :param x: [N,E] FloatTensor
        :return:
        """
        return self.model(x)
