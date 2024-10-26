# -*- coding: utf-8 -*-
import json
import os

import torch

from ..datas.tokenizer import MyTokenizer
from ..models.bert_crf_ner import BertCrfNerModel
from ..utils import trans_entity2tuple


class Predictor(object):
    def __init__(self, model_dir):
        super(Predictor, self).__init__()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        model_path = os.path.join(model_dir, "best_model.pt")
        config_path = os.path.join(model_dir, "config.json")
        dict_path = os.path.join(model_dir, "vocab.txt")
        category_path = os.path.join(model_dir, "categories.json")

        with open(category_path, 'r', encoding="utf-8") as reader:
            categories = json.load(reader)

        self.model = BertCrfNerModel(
            config_path=config_path,
            checkpoint_path=None,
            num_tags=len(categories)
        ).to(device)
        self.max_len = self.model.bert.max_position
        self.model.eval()
        self.model.load_weights(model_path)  # 模型参数恢复加载
        self.tokenizer = MyTokenizer(dict_path, do_lower_case=True)
        self.categories_id2label = {i: k for i, k in enumerate(categories)}
        print("预测器恢复完成!")

    def _predict(self, text, global_start_pos=0):
        """
        :param text: text的长度一定是小于max_len - 2
        :param global_start_pos:
        :return:
        """
        tokens = self.tokenizer.tokenize(text, maxlen=self.max_len)
        token_ids = self.tokenizer.tokens_to_ids(tokens)
        token_ids = torch.tensor([token_ids], dtype=torch.int64, device=self.device)
        scores = self.model.predict(token_ids)  # [btz, seq_len]
        print(json.dumps(list(map(int, scores.numpy()[0]))))
        entity_pred = trans_entity2tuple(scores, self.categories_id2label)  # CRF的解析，得到最终的预测结果
        entity_pred = list(entity_pred)
        entity_pred = sorted(entity_pred, key=lambda t: t[1])
        print(json.dumps(list(entity_pred), ensure_ascii=False))

        # 结果拼接
        entities = []
        for _, i, j, label in entity_pred:
            entity_span = text[i - 1:j]  # i-1原因是前面加入了CLS前缀
            entities.append({
                "label_type": label,
                "start_pos": i - 1 + global_start_pos,  # 包含start_pos
                "end_pos": j + global_start_pos,  # 不包含end_pos,
                "span": entity_span
            })

        return entities

    @torch.no_grad()
    def predict(self, text: str):
        """
        针对文本进行预测
        :param text:
        :return:
        """
        text_len = len(text)
        entities = []
        if text_len < self.max_len:
            entities.extend(self._predict(text))
        else:
            len_per_bucket = text_len // (text_len // (self.max_len - 2) + 1)  # 每个区间的预估长度
            start_pos = 0
            while start_pos < text_len:
                # print(start_pos, text_len)
                sub_text = text[start_pos: start_pos + len_per_bucket]
                if start_pos + len_per_bucket < text_len:
                    for i in range(len(sub_text) - 1, -1, -1):
                        sub_char = sub_text[i]
                        if sub_char in [',', '.', '，', '。']:
                            sub_text = sub_text[0:i]
                            break
                entities.extend(self._predict(sub_text, global_start_pos=start_pos))
                start_pos = start_pos + len(sub_text)

        # 排序
        entities.sort(key=lambda t: t['start_pos'])

        return {
            'originalText': text,
            'entities': entities
        }
