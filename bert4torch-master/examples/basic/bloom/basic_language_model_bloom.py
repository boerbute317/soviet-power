#! -*- coding: utf-8 -*-
"""
基本测试：bloom模型的测试
bert4torch_config.json文件参考readme

bloom-560m: https://huggingface.co/bigscience/bloom-560m
bloomz-560m:  https://huggingface.co/bigscience/bloomz-560m
"""

import torch
from bert4torch.models import build_transformer_model
from bert4torch.generation import SeqGeneration
from transformers import AutoTokenizer


model_dir = '/data/pretrain_ckpt/bloom/bloomz-560m'  # bloom-560m  bloomz-560m
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
model = build_transformer_model(config_path=model_dir, checkpoint_path=model_dir)
# model = model.quantize(quantization_method='cpm_kernels', quantization_bit=8)
model = model.to(device)

generation = SeqGeneration(model, tokenizer, start_id=None, end_id=tokenizer.eos_token_id,
                           tokenizer_config={'skip_special_tokens': True})


if __name__ == '__main__':
    while True:
        query = input("\n输入：")
        response = generation.generate(query, topk=1, include_input=True)      
        print(f"续写:{response}")