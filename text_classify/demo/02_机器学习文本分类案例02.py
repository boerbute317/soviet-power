# -*- coding: utf-8 -*-
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import classification_report


def t0():
    remove = ()
    categories = [
        "alt.atheism",
        "talk.religion.misc",
        "comp.graphics",
        "sci.space",
    ]

    # 1. 数据加载
    data_train = fetch_20newsgroups(
        subset="train",
        categories=categories,
        shuffle=True,
        random_state=42,
        remove=remove,
    )

    data_test = fetch_20newsgroups(
        subset="test",
        categories=categories,
        shuffle=True,
        random_state=42,
        remove=remove,
    )

    # 2. 数据预处理：异常数据的过滤、缺失数据的填充、数据去重....
    # 3. 特征工程：
    #         针对离散特征进行OneHot哑编码处理
    #         连续特征离散化/分区/分桶处理
    #         标准化
    #         归一化
    vectorizer = TfidfVectorizer(
        sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english"
    )
    X_train = vectorizer.fit_transform(data_train.data)
    X_test = vectorizer.transform(data_test.data)
    y_train, y_test = data_train.target, data_test.target

    # 4. 模型对象的创建(算法结构的确定、优化器的确定)
    clf = RidgeClassifier(tol=1e-2, solver="sparse_cg")

    # 5. 模型的训练优化
    clf.fit(X_train, y_train)  # X_train一定是一个矩阵的特征属性结构[N,E] 表示N个样本，每个样本E维的一个向量

    # 6. 模型评估
    pred = clf.predict(X_test)
    report = classification_report(y_test, pred)
    print(report)

    # 7. 模型持久化
    pass


if __name__ == '__main__':
    t0()
