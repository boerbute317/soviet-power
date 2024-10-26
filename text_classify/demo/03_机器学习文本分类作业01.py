# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import jieba

FILE_ROOT_DIR = os.path.dirname(__file__)


# %%

def t0():
    # 1. 加载数据
    print(f"分类作业01文件输入当前根目录:{os.getcwd()}")
    print(f"当前文件所在文件夹路径:{FILE_ROOT_DIR}")
    df = pd.read_csv(
        os.path.join(FILE_ROOT_DIR, "../datas/train.csv"),
        header=None, names=['x', 'y'], sep="\t"
    )
    X = df.x.values
    X = [' '.join(jieba.lcut(x)) for x in X]
    Y = df.y.values
    y_labels = np.unique(Y)
    ylabel2idx = dict(zip(y_labels, range(len(y_labels))))
    Y = [ylabel2idx[y] for y in Y]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    # 2.数据预处理 & 3.特征工程
    vectorizer = TfidfVectorizer(
        sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english"
    )
    # 输入是list[str] 输出是特征属性矩阵,shape为[N,E]
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # 4. 模型对象的创建(算法结构的确定、优化器的确定)
    clf = RidgeClassifier(tol=1e-2, solver="sparse_cg")

    # 5. 模型的训练优化
    clf.fit(X_train, y_train)  # X_train一定是一个矩阵的特征属性结构[N,E] 表示N个样本，每个样本E维的一个向量

    # 6. 模型评估
    pred = clf.predict(X_test)
    report = classification_report(y_test, pred)
    print(report)


# %%

if __name__ == '__main__':
    t0()
