# -*- coding: utf-8 -*-
import os

from torch4keras.callbacks import Callback
from tqdm import tqdm

from ..utils import trans_entity2tuple


def evaluate(model, data, categories_id2label):
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
        entity_pred = trans_entity2tuple(scores, categories_id2label)
        entity_true = trans_entity2tuple(label, categories_id2label)
        X2 += len(entity_pred.intersection(entity_true))
        Y2 += len(entity_pred)
        Z2 += len(entity_true)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    f2, precision2, recall2 = 2 * X2 / (Y2 + Z2), X2 / Y2, X2 / Z2
    return f1, precision, recall, f2, precision2, recall2


class Evaluator(Callback):
    """评估与保存
    """

    def __init__(self, model, data, categories_id2label, model_save_dir):
        super(Evaluator, self).__init__()
        self.best_val_f1 = 0.
        self.model = model
        self.data = data
        self.categories_id2label = categories_id2label
        self.model_save_dir = model_save_dir
        os.makedirs(self.model_save_dir, exist_ok=True)

    def on_epoch_end(self, steps, epoch, logs=None):
        f1, precision, recall, f2, precision2, recall2 = evaluate(
            self.model, self.data, self.categories_id2label
        )
        if f2 > self.best_val_f1:
            self.best_val_f1 = f2
            self.model.save_weights(os.path.join(self.model_save_dir, 'best_model.pt'))
        print(
            f'[val-token  level] f1: {f1:.5f}, p: {precision:.5f} r: {recall:.5f}'
        )
        print(
            f'[val-entity level] f1: {f2:.5f}, p: {precision2:.5f} r: {recall2:.5f} best_f1: {self.best_val_f1:.5f}\n'
        )
