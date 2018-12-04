import json
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
from gensim import corpora, matutils


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
    vec /= np.linalg.norm(vec)
    return vec

# 正規化されたベクトルのコサイン類似度を求める
def cos_sim(v1, v2):
    # return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.dot(v1, v2)

# srcの中からtargetと最もコサイン類似度の高い要素のラベルを返す
def most_sim(target, src):
    # max = -1
    max = 0
    idx = 0
    for i, v in enumerate(src):
        sim = cos_sim(v['vec'], target['vec'])
        if sim > max:
            max = sim
            idx = i
    return src[idx]['label']


if __name__ == '__main__':
    dataset_dir = './static/processed/v6/'

    norm_train = list(read_dataset(dataset_dir+'norm-train.jsonl'))
    anom_train = list(read_dataset(dataset_dir+'anom-train.jsonl'))
    norm_test = list(read_dataset(dataset_dir+'norm-test.jsonl'))
    anom_test = list(read_dataset(dataset_dir+'anom-test.jsonl'))

    # trainデータから単語辞書を生成
    dct = corpora.Dictionary(norm_train + anom_train)

    # trainベクトルを生成
    norm_train = [{'vec': words2vec(dct, v), 'label': 'norm'} for v in norm_train]
    anom_train = [{'vec': words2vec(dct, v), 'label': 'anom'} for v in anom_train]

    # testベクトルを生成
    norm_test = [{'vec': words2vec(dct, v), 'label': 'norm'} for v in norm_test]
    anom_test = [{'vec': words2vec(dct, v), 'label': 'anom'} for v in anom_test]

    # 最近傍法でテスト　
    src = norm_train + anom_train
    TP = FN = FP = TN = 0
    for v in tqdm(norm_test):
        if most_sim(v, src) == 'norm':
            TP += 1
        else:
            FN += 1
    for v in tqdm(anom_test):
        if most_sim(v, src) == 'norm':
            FP += 1
        else:
            TN += 1

    Accuracy = (TP + TN) / (TP + FP + FN + TN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = 2 * Recall * Precision / (Recall + Precision)

    print('Accuracy:  {}\n'.format(Accuracy))
    print('Precision: {}\n'.format(Precision))
    print('Recall:    {}\n'.format(Recall))
    print('F1:        {}\n'.format(F1))