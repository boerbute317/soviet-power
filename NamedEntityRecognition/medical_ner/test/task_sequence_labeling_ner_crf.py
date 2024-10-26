#! -*- coding:utf-8 -*-
# bert+crf用来做实体识别
# 数据集：http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz
# [valid_f1]  token_level: 97.06； entity_level: 95.90
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from bert4torch.callbacks import Callback
from bert4torch.snippets import sequence_padding, ListDataset, seed_everything
from bert4torch.layers import CRF
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from tqdm import tqdm

maxlen = 256
batch_size = 8
# 数据采用的标注方式是:BIO B表示这个token是某个实体的开头，I表示这个token是某个实体的中间或者结尾 O表示token不属于实体
categories = ['O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG']
categories_id2label = {i: k for i, k in enumerate(categories)}
categories_label2id = {k: i for i, k in enumerate(categories)}
data_root_path = "./datas/china-people-daily-ner-corpus"

# BERT base
# bert-base-chinese分享路径：链接：https://pan.baidu.com/s/1_MzfBLo_JLZllWqR7Lxu3w?pwd=h2oo  提取码：h2oo
bert_root_path = r"D:\huggingface\huggingface\hub\models--bert-base-chinese"
config_path = os.path.join(bert_root_path, "config.json")
checkpoint_path = os.path.join(bert_root_path, "pytorch_model.bin")
dict_path = os.path.join(bert_root_path, "vocab.txt")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 固定seed 设置随机数种子
seed_everything(42)


# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        D = []
        with open(filename, encoding='utf-8') as f:
            f = f.read()  # 加载所有数据
            for l in f.split('\n\n'):
                if not l:
                    continue
                d = ['']
                for i, c in enumerate(l.split('\n')):
                    char, flag = c.split(' ')  # token以及对应的标签
                    d[0] += char
                    if flag[0] == 'B':
                        d.append([i, i, flag[2:]])  # 从i->i属于实体flag[2:]
                    elif flag[0] == 'I':
                        d[-1][1] = i
                D.append(d)
        return D


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def collate_fn(batch):
    batch_token_ids, batch_labels = [], []
    for d in batch:
        tokens = tokenizer.tokenize(d[0], maxlen=maxlen)  # 文本分词/分字 + 特殊token处理
        mapping = tokenizer.rematch(d[0], tokens)
        start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
        end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
        token_ids = tokenizer.tokens_to_ids(tokens)
        labels = np.zeros(len(token_ids))
        for start, end, label in d[1:]:
            if start in start_mapping and end in end_mapping:
                start = start_mapping[start]  # 原始的开始token位置 映射 现在的token位置
                end = end_mapping[end]  # 原始的结束token位置 映射 现在的token位置
                labels[start] = categories_label2id['B-' + label]
                labels[start + 1:end + 1] = categories_label2id['I-' + label]
        batch_token_ids.append(token_ids)
        batch_labels.append(labels)
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(sequence_padding(batch_labels), dtype=torch.long, device=device)
    return batch_token_ids, batch_labels


# 转换数据集
# noinspection PyTypeChecker
train_dataloader = DataLoader(
    # MyDataset(os.path.join(data_root_path, 'example.train')),
    MyDataset(os.path.join(data_root_path, 'min_example.train')),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)
# noinspection PyTypeChecker
valid_dataloader = DataLoader(
    # MyDataset(os.path.join(data_root_path, 'example.dev')),
    MyDataset(os.path.join(data_root_path, 'min_example.dev')),
    batch_size=batch_size,
    collate_fn=collate_fn
)


# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.bert = build_transformer_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            segment_vocab_size=0,
            model='bert'
        )
        self.fc = nn.Linear(768, len(categories))  # 包含首尾
        self.crf = CRF(len(categories))  # 在softmax的基础上额外增加一个类别到类别的转换概率矩阵

    def forward(self, token_ids):
        sequence_output = self.bert([token_ids])  # [btz, seq_len, hdsz]
        emission_score = self.fc(sequence_output)  # [btz, seq_len, tag_size]
        attention_mask = token_ids.gt(0).long()  # 构建mask矩阵
        return emission_score, attention_mask

    def predict(self, token_ids):
        self.eval()
        with torch.no_grad():
            emission_score, attention_mask = self.forward(token_ids)
            best_path = self.crf.decode(emission_score, attention_mask)  # [btz, seq_len]
        return best_path


model = Model().to(device)


class Loss(nn.Module):
    def forward(self, outputs, labels):
        return model.crf(*outputs, labels)  # 计算CRF的损失


def acc(y_pred, y_true):
    y_pred = y_pred[0]
    y_pred = torch.argmax(y_pred, dim=-1)
    acc = torch.sum(y_pred.eq(y_true)).item() / y_true.numel()
    return {'acc': acc}


def acc2(y_pred, y_true):
    emission_score, attention_mask = y_pred
    # noinspection PyUnresolvedReferences
    y_pred = model.crf.decode(emission_score, attention_mask).to(y_true.dtype)  # 预测结果/标签的提取
    acc = torch.sum(y_pred.eq(y_true)).item() / y_true.numel()
    return {'acc2': acc}


def acc3(y_pred, y_true):
    emission_score, attention_mask = y_pred
    # noinspection PyUnresolvedReferences
    y_pred = model.crf.decode(emission_score, attention_mask).to(y_true.dtype)  # 预测结果/标签的提取

    y_true_mask = y_true.gt(0).float()
    total_y_true_tag = y_true_mask.sum().item()
    if total_y_true_tag > 0:
        acc = torch.sum(y_pred.eq(y_true).float() * y_true_mask).item() / total_y_true_tag
    else:
        acc = -1.0
    return {'acc3': acc}


# 支持多种自定义metrics = ['accuracy', acc, {acc: acc}]均可
model.compile(
    loss=Loss(),
    optimizer=optim.Adam(model.parameters(), lr=2e-5),
    metrics=[acc, acc2, acc3]
)


def evaluate(data):
    X, Y, Z = 1e-10, 1e-10, 1e-10
    X2, Y2, Z2 = 1e-10, 1e-10, 1e-10
    for token_ids, label in tqdm(data):
        scores = model.predict(token_ids)  # [btz, seq_len]
        attention_mask = label.gt(0)

        # token粒度
        X += (scores.eq(label) * attention_mask).sum().item()
        Y += scores.gt(0).sum().item()
        Z += label.gt(0).sum().item()

        # entity粒度
        entity_pred = trans_entity2tuple(scores)
        entity_true = trans_entity2tuple(label)
        X2 += len(entity_pred.intersection(entity_true))
        Y2 += len(entity_pred)
        Z2 += len(entity_true)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    f2, precision2, recall2 = 2 * X2 / (Y2 + Z2), X2 / Y2, X2 / Z2
    return f1, precision, recall, f2, precision2, recall2


def trans_entity2tuple(scores):
    '''把tensor转为(样本id, start, end, 实体类型)的tuple用于计算指标
    '''
    batch_entity_ids = set()
    for i, one_samp in enumerate(scores):
        entity_ids = []
        for j, item in enumerate(one_samp):
            flag_tag = categories_id2label[item.item()]
            if flag_tag.startswith('B-'):  # B
                entity_ids.append([i, j, j, flag_tag[2:]])
            elif len(entity_ids) == 0:
                continue
            elif (len(entity_ids[-1]) > 0) and flag_tag.startswith('I-') and (flag_tag[2:] == entity_ids[-1][-1]):  # I
                entity_ids[-1][-2] = j
            elif len(entity_ids[-1]) > 0:
                entity_ids.append([])

        for i in entity_ids:
            if i:
                batch_entity_ids.add(tuple(i))
    return batch_entity_ids


class Evaluator(Callback):
    """评估与保存
    """

    def __init__(self):
        super(Evaluator, self).__init__()
        self.best_val_f1 = 0.

    def on_epoch_end(self, steps, epoch, logs=None):
        f1, precision, recall, f2, precision2, recall2 = evaluate(valid_dataloader)
        if f2 > self.best_val_f1:
            self.best_val_f1 = f2
            model.save_weights('best_model.pt')
        print(
            f'[val-token  level] f1: {f1:.5f}, p: {precision:.5f} r: {recall:.5f}'
        )
        print(
            f'[val-entity level] f1: {f2:.5f}, p: {precision2:.5f} r: {recall2:.5f} best_f1: {self.best_val_f1:.5f}\n'
        )


if __name__ == '__main__':
    print("执行训练代码逻辑....")
    evaluator = Evaluator()
    model.fit(
        train_dataloader,
        epochs=20,
        steps_per_epoch=None,
        callbacks=[evaluator]
    )
else:
    print("执行模型参数恢复代码逻辑....")
    model.load_weights('best_model.pt')
