import json
import warnings
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
from gensim import corpora, matutils
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# jsonl形式のテスト用データセットを読み込む
def read_dataset(path):
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            yield d['words']

# 単語配列をBoWでベクトル化
def words2vec(dct, words):
    b = dct.doc2bow(words)
    vec = np.array(list(matutils.corpus2dense([b], num_terms=len(dct)).T[0]))
    # 正規化
    # vec /= np.linalg.norm(vec)
    return vec


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=FutureWarning)

    dataset_dir = './static/processed/v6/'

    norm_train = list(read_dataset(dataset_dir+'norm-train.jsonl'))
    anom_train = list(read_dataset(dataset_dir+'anom-train.jsonl'))
    norm_test = list(read_dataset(dataset_dir+'norm-test.jsonl'))
    anom_test = list(read_dataset(dataset_dir+'anom-test.jsonl'))

    # trainデータから単語辞書を生成
    dct = corpora.Dictionary(norm_train + anom_train)

    # trainベクトルを生成
    norm_train = [words2vec(dct, v) for v in norm_train]
    anom_train = [words2vec(dct, v) for v in anom_train]

    # testベクトルを生成
    norm_test = [words2vec(dct, v) for v in norm_test]
    anom_test = [words2vec(dct, v) for v in anom_test]

    # ランダムフォレストでテスト　
    x_train = norm_train + anom_train
    y_train = ['norm']*len(norm_train) + ['anom']*len(anom_train)

    x_test = norm_test + anom_test
    y_test = ['norm']*len(norm_test) + ['anom']*len(anom_test)

    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    Accuracy = accuracy_score(y_test, y_pred)
    Precision = precision_score(y_test, y_pred, pos_label='norm')
    Recall = precision_score(y_test, y_pred, pos_label='norm')

    print('Accuracy:  {}\n'.format(Accuracy))
    print('Precision: {}\n'.format(Precision))
    print('Recall:    {}\n'.format(Recall))