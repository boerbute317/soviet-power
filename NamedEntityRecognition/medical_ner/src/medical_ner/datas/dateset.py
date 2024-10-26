# -*- coding: utf-8 -*-
import json
from typing import Union, Optional

import numpy as np
import pandas as pd
import torch

from torch4keras.snippets import ListDataset

from bert4torch.snippets import sequence_padding


class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        D = []
        with open(filename, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                D.append(json.loads(line))
        return D

    @staticmethod
    def build_collect_fn(tokenizer, maxlen, categories_label2id, device):
        def collate_fn(batch):
            batch_token_ids, batch_labels = [], []
            for d in batch:
                tokens = tokenizer.tokenize(d['originalText'], maxlen=maxlen)  # 文本分词/分字 + 特殊token处理
                token_ids = tokenizer.tokens_to_ids(tokens)
                labels = np.zeros(len(token_ids))
                for entity in d['entities']:
                    start, end, label = entity['start_pos'], entity['end_pos'], entity['label_type']
                    if end < len(tokens):
                        labels[start + 1] = categories_label2id['B-' + label]
                        labels[start + 2:end + 1] = categories_label2id['I-' + label]
                batch_token_ids.append(token_ids)
                batch_labels.append(labels)
            batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
            batch_labels = torch.tensor(sequence_padding(batch_labels), dtype=torch.long, device=device)
            return batch_token_ids, batch_labels

        return collate_fn


class MyTravelQueryDataset(ListDataset):
    def __init__(self,
                 tokenizer, categories_label2id, maxlen,
                 file_path: Union[str, tuple, list] = None, data: Optional[list] = None
                 ):
        self.tokenizer = tokenizer
        self.categories_label2id = categories_label2id
        self.maxlen = maxlen
        super().__init__(file_path, data)

    def load_data(self, filename):
        df = pd.read_excel(filename)
        df.fillna(value='', inplace=True)

        datas = []
        x = []
        y = []

        def _add_record():
            tokens = self.tokenizer.tokenize(''.join(x), maxlen=self.maxlen)  # 文本分词/分字 + 特殊token处理
            token_ids = self.tokenizer.tokens_to_ids(tokens)
            labels = [0] + y + [0]
            datas.append((token_ids, labels))

        for token, tag in df.values:
            token = token.strip()
            if len(token) == 0:
                if len(x) == 0:
                    continue
                # 表示新的一个样本
                _add_record()
                x = []
                y = []
            else:
                x.append(token)
                if 'O' == tag:
                    tag = tag
                elif tag.startswith("B-") or tag.startswith("S-"):
                    tag = "B-" + tag[2:]
                else:
                    tag = "I-" + tag[2:]
                y.append(self.categories_label2id[tag])
        if len(x) > 0:
            _add_record()

        return datas

    @staticmethod
    def build_collect_fn(device):
        def collate_fn(batch):
            batch_token_ids, batch_labels = [], []
            for token_ids, labels in batch:
                batch_token_ids.append(token_ids)
                batch_labels.append(labels)
            batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
            batch_labels = torch.tensor(sequence_padding(batch_labels), dtype=torch.long, device=device)
            return batch_token_ids, batch_labels

        return collate_fn
