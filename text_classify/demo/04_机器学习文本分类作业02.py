# -*- coding: utf-8 -*-

import os
import jieba
import numpy as np
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import classification_report
from sklearn import metrics

FILE_ROOT_DIR = os.path.dirname(__file__)

# %% 数据预处理: x的分词、y的标签转换

jieba.load_userdict(os.path.join(FILE_ROOT_DIR, "../datas/jieba.words"))

df = pd.read_csv(
    os.path.join(FILE_ROOT_DIR, "../datas/train.csv"),
    header=None, names=['x', 'y'], sep="\t"
)

# 分词 + 标签转换
df['cut_x'] = [' '.join(jieba.lcut(x)) for x in df.x.values]
Y = df.y.values
y_labels = np.unique(Y)
ylabel2idx = dict(zip(y_labels, range(len(y_labels))))
df['y_label_idx'] = [ylabel2idx[y] for y in Y]

# 输出各个类别单独一个csv文件
category_output_dir = os.path.join(FILE_ROOT_DIR, "../datas/each_category")
os.makedirs(category_output_dir, exist_ok=True)
for y_label, group_df in df.groupby('y'):
    group_df.to_csv(os.path.join(category_output_dir, f"{y_label}.csv"), index=False, sep="\t")

with open(os.path.join(FILE_ROOT_DIR, "../datas/label2idx.json"), "w", encoding="utf-8") as writer:
    json.dump(ylabel2idx, writer, indent=2)

# 输出最终参与模型训练的数据结果
df[['cut_x', 'y_label_idx']].to_csv(
    os.path.join(FILE_ROOT_DIR, "../datas/split_train.csv")
    , header=None, index=False, sep="\t"
)

# %% 参与训练的处理后的数据加载

df = pd.read_csv(os.path.join(FILE_ROOT_DIR, "../datas/split_train.csv"), sep="\t", header=None, names=['x', 'y'])
X_train, X_test, y_train, y_test = train_test_split(df.x.values, df.y.values, test_size=0.2, random_state=28)
with open(os.path.join(FILE_ROOT_DIR, "../datas/label2idx.json"), "r", encoding="utf-8") as reader:
    ylabel2idx = json.load(reader)
    yidx2label = {v: k for k, v in ylabel2idx.items()}

# %% 文本向量转换

vectorizer = TfidfVectorizer(
    sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english"
)
# 输入是list[str] 输出是特征属性矩阵,shape为[N,E]
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# %% 模型构建

clf = RidgeClassifier(tol=1e-2, solver="sparse_cg")

# %%  模型的训练优化

clf.fit(X_train_vec, y_train)  # X_train一定是一个矩阵的特征属性结构[N,E] 表示N个样本，每个样本E维的一个向量

# %% 模型评估

pred = clf.predict(X_test_vec)
report = classification_report(y_test, pred)
print(report)

confusion_matrix = metrics.confusion_matrix(y_test, pred)
print(confusion_matrix)

# %% 将预测结果和实际结果比较输出(一般情况下输出不一样的数据即可)

y_test_label = [yidx2label[i] for i in y_test]
pred_label = [yidx2label[i] for i in pred]
pred_df = pd.DataFrame([X_test, y_test, pred, y_test_label, pred_label]).T
pred_df.columns = ['分词后的文本', '真实标签id', '预测标签id', '真实标签', '预测标签']
bad_case_df = pred_df[pred_df.真实标签id != pred_df.预测标签id]
bad_case_output_dir = os.path.join(FILE_ROOT_DIR, "../datas/base_case")
os.makedirs(bad_case_output_dir, exist_ok=True)
for y_label, bad_case_group_df in bad_case_df.groupby('真实标签'):
    bad_case_group_df.sort_values("预测标签", inplace=True)
    bad_case_group_df.to_csv(os.path.join(bad_case_output_dir, f"{y_label}.csv"), index=False, sep="\t")

# %% bad case分析


text = [
    '周五 是 中秋 吗'
]
text_vec = vectorizer.transform(text).toarray()

print((text_vec != 0).sum())
print(vectorizer.inverse_transform(text_vec))
print(clf.predict(text_vec))
log_proba = clf.decision_function(text_vec)
print(log_proba)
print(log_proba.argsort()[:, ::-1])