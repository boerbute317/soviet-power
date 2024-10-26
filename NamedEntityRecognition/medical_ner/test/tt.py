# -*- coding: utf-8 -*-

import torch

ckpt = torch.load('./ckpt/best_model.pt', map_location='cpu')
print(type(ckpt))
print(ckpt.keys())
