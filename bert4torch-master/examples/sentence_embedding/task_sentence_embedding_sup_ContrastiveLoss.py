#! -*- coding:utf-8 -*-
# loss: ContrastiveLoss

from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.callbacks import Callback
from bert4torch.snippets import sequence_padding, ListDataset, get_pool_emb, seed_everything
from bert4torch.losses import ContrastiveLoss
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import paired_cosine_distances
from scipy.stats import spearmanr
from tqdm import tqdm
import argparse
import numpy as np


# =============================基本参数=============================
parser = argparse.ArgumentParser()
parser.add_argument('--pooling', default='cls', choices=['first-last-avg', 'last-avg', 'cls', 'pooler'])
parser.add_argument('--task_name', default='ATEC', choices=['ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STS-B'])
args = parser.parse_args()
pooling = args.pooling
task_name = args.task_name

maxlen = 64 if task_name != 'PAWSX' else 128
batch_size = 32
config_path = 'E:/pretrain_ckpt/bert/google@chinese_L-12_H-768_A-12/bert4torch_config.json'
checkpoint_path = 'E:/pretrain_ckpt/bert/google@chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'E:/pretrain_ckpt/bert/google@chinese_L-12_H-768_A-12/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed_everything(42)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        """加载数据
        单条格式：(文本1, 文本2, 标签id)
        """
        D = []
        with open(filename, encoding='utf-8') as f:
            for l in f:
                l = l.strip().split('\t')
                if len(l) == 3:
                    D.append((l[0], l[1], int(l[2])))
        return D

def collate_fn(batch):
    batch_token1_ids, batch_token2_ids, batch_labels = [], [], []
    for text1, text2, label in batch:
        token1_ids, _ = tokenizer.encode(text1, maxlen=maxlen)
        batch_token1_ids.append(token1_ids)
        token2_ids, _ = tokenizer.encode(text2, maxlen=maxlen)
        batch_token2_ids.append(token2_ids)
        batch_labels.append([int(label>2.5) if task_name == 'STS-B' else label])

    batch_token1_ids = torch.tensor(sequence_padding(batch_token1_ids), dtype=torch.long, device=device)
    batch_token2_ids = torch.tensor(sequence_padding(batch_token2_ids), dtype=torch.long, device=device)

    batch_labels = torch.tensor(batch_labels, dtype=torch.float, device=device)
    return (batch_token1_ids, batch_token2_ids), batch_labels.flatten()

# 加载数据集
train_dataloader = DataLoader(MyDataset(f'F:/data/corpus/sentence_embedding/{task_name}/{task_name}.train.data'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataloader = DataLoader(MyDataset(f'F:/data/corpus/sentence_embedding/{task_name}/{task_name}.valid.data'), batch_size=batch_size, collate_fn=collate_fn)
test_dataloader = DataLoader(MyDataset(f'F:/data/corpus/sentence_embedding/{task_name}/{task_name}.test.data'), batch_size=batch_size, collate_fn=collate_fn)

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self, pool_method='cls'):
        super().__init__()
        self.pool_method = pool_method
        with_pool = 'linear' if pool_method == 'pooler' else True
        output_all_encoded_layers = True if pool_method == 'first-last-avg' else False
        self.bert = build_transformer_model(config_path, checkpoint_path, segment_vocab_size=0,
                                            with_pool=with_pool, output_all_encoded_layers=output_all_encoded_layers)

    def forward(self, token1_ids, token2_ids):
        hidden_state1, pool_cls1 = self.bert([token1_ids])
        pool_emb1 = get_pool_emb(hidden_state1, pool_cls1, token1_ids.gt(0).long(), self.pool_method)
        
        hidden_state2, pool_cls2 = self.bert([token2_ids])
        pool_emb2 = get_pool_emb(hidden_state2, pool_cls2, token2_ids.gt(0).long(), self.pool_method)

        distance = 1- torch.cosine_similarity(pool_emb1, pool_emb2)
        return distance
    
    def predict(self, token_ids):
        self.eval()
        with torch.no_grad():
            hidden_state, pooler = self.bert([token_ids])
            attention_mask = token_ids.gt(0).long()
            output = get_pool_emb(hidden_state, pooler, attention_mask, self.pool_method)
        return output

model = Model().to(device)

# 定义使用的loss和optimizer，这里支持自定义
model.compile(
    loss=ContrastiveLoss(),
    optimizer=optim.Adam(model.parameters(), lr=2e-5),
)

class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_consine = 0.

    def on_epoch_end(self, global_step, epoch, logs=None):
        val_consine = self.evaluate(valid_dataloader)
        test_consine = self.evaluate(test_dataloader)

        if val_consine > self.best_val_consine:
            self.best_val_consine = val_consine
            # model.save_weights('best_model.pt')
        print(f'valid_consine: {val_consine:.5f}, test_consine: {test_consine:.5f}, best_val_consine: {self.best_val_consine:.5f}\n')

    # 定义评价函数
    def evaluate(self, data):
        cosine_scores, labels = [], []
        for (batch_token1_ids, batch_token2_ids), batch_labels in tqdm(data, desc='Evaluate'):
            embeddings1 = model.predict(batch_token1_ids).cpu()
            embeddings2 = model.predict(batch_token2_ids).cpu()
            cosine_score = 1 - paired_cosine_distances(embeddings1, embeddings2)
            cosine_scores.append(cosine_score)
            labels.append(batch_labels.cpu().numpy())
        labels = np.concatenate(labels)
        cosine_scores = np.concatenate(cosine_scores)
        eval_pearson_cosine, _ = spearmanr(labels, cosine_scores)
        return eval_pearson_cosine


if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit(train_dataloader, epochs=5, steps_per_epoch=None, callbacks=[evaluator])
else:
    model.load_weights('best_model.pt')
