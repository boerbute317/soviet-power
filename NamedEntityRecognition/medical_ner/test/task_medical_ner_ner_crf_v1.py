#! -*- coding:utf-8 -*-
# bert+crf用来做实体识别
# 数据集：http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz
# [valid_f1]  token_level: 97.06； entity_level: 95.90
import json
import os

import torch.optim as optim
from torch.utils.data import DataLoader

from bert4torch.snippets import seed_everything
from medical_ner.callback.evaluator import Evaluator
from medical_ner.datas.dateset import MyDataset
from medical_ner.datas.tokenizer import MyTokenizer
from medical_ner.metrices.accuracy import *
from medical_ner.models.bert_crf_ner import BertCrfNerModel
from medical_ner.models.bert_softmax_ner import BertSoftmaxNerModel

if __name__ == '__main__':
    print("执行训练代码逻辑....")

    data_root_path = "./datas/medical"
    model_save_dir = "."
    maxlen = 512
    batch_size = 4
    # 数据采用的标注方式是:BIO B表示这个token是某个实体的开头，I表示这个token是某个实体的中间或者结尾 O表示token不属于实体
    with open(os.path.join(data_root_path, "categories.json"), 'r', encoding="utf-8") as reader:
        categories = json.load(reader)
    categories_id2label = {i: k for i, k in enumerate(categories)}
    categories_label2id = {k: i for i, k in enumerate(categories)}

    # BERT base
    # bert-base-chinese分享路径：链接：https://pan.baidu.com/s/1_MzfBLo_JLZllWqR7Lxu3w?pwd=h2oo  提取码：h2oo
    if 'nt' == os.name:
        bert_root_path = r"D:\huggingface\huggingface\hub\models--bert-base-chinese"
    else:
        bert_root_path = r"/root/workspace/models--bert-base-chinese"
    config_path = os.path.join(bert_root_path, "config.json")
    checkpoint_path = os.path.join(bert_root_path, "pytorch_model.bin")
    dict_path = os.path.join(bert_root_path, "vocab.txt")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 固定seed 设置随机数种子
    seed_everything(42)

    # 建立分词器
    tokenizer = MyTokenizer(dict_path, do_lower_case=True)
    collate_fn = MyDataset.build_collect_fn(tokenizer, maxlen, categories_label2id, device)

    # 转换数据集
    # noinspection PyTypeChecker
    train_dataloader = DataLoader(
        MyDataset(os.path.join(data_root_path, 'training.txt')),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    # noinspection PyTypeChecker
    valid_dataloader = DataLoader(
        MyDataset(os.path.join(data_root_path, 'test.json')),
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    # 定义bert上的模型结构
    model = BertCrfNerModel(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        num_tags=len(categories)
    ).to(device)

    # 如果你的模型使用了CRF，那么CRF部分的参数最好学习率大一点 --> 所以导致CRF部分参数不好学习
    # NOTE: 当你的数据足够多、质量足够好的时候，不使用CRF也可以
    param_g0, param_g1 = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'crf' in name:
            param_g1.append(param)
        else:
            param_g0.append(param)
    optimizer = optim.Adam(param_g0, lr=2e-5)
    if len(param_g1) > 0:
        print(f"CRF参数数量:{len(param_g1)}")
        optimizer.add_param_group({'params': param_g1, 'lr': 0.1})

    # 支持多种自定义metrics = ['accuracy', acc, {acc: acc}]均可
    model.compile(
        loss=model.build_loss_fn(),
        optimizer=optimizer,
        metrics=[acc, build_acc2(model), build_acc3(model)]
    )

    evaluator = Evaluator(
        model=model,
        data=valid_dataloader,
        categories_id2label=categories_id2label,
        model_save_dir=model_save_dir
    )
    model.fit(
        train_dataloader,
        epochs=20,
        steps_per_epoch=None,
        callbacks=[evaluator]
    )
