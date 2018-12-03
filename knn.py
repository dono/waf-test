import json
import random
import warnings
import numpy as np
from concurrent import futures
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# jsonl形式のトレーニング用データセットを読み込む
def read_train_dataset(path):
    with open(path) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            yield TaggedDocument(d['words'], [d['label']+str(i)])

# jsonl形式のテスト用データセットを読み込む
def read_test_dataset(path):
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            yield d


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    dataset_dir = './static/processed/v1/'

    norm_train = list(read_train_dataset(dataset_dir + 'norm-train.jsonl'))
    anom_train = list(read_train_dataset(dataset_dir + 'anom-train.jsonl'))

    # model = Doc2Vec(norm_train + anom_train, dm=1, window=2, min_count=1, vector_size=600, alpha=0.08, min_alpha=0.01, epochs=600, workers=6)
    # model.save('./tmp/doc2vec.model')
    model = Doc2Vec.load('./tmp/doc2vec.model')

    norm_train_vecs = [model.docvecs['norm'+str(i)] for i in range(len(norm_train))]
    anom_train_vecs = [model.docvecs['anom'+str(i)] for i in range(len(anom_train))]

    x_train = norm_train_vecs + anom_train_vecs
    y_train = ['norm']*len(norm_train_vecs) + ['anom']*len(anom_train_vecs)

    clf = KNeighborsClassifier(n_neighbors=1, metric='cosine', n_jobs=-1)
    clf.fit(x_train, y_train)

    # testing
    norm_test_vecs = []
    norms = list(read_test_dataset(dataset_dir + 'norm-test.jsonl'))
    for norm in norms:
        norm_test_vecs.append(model.infer_vector(norm['words']))

    anom_test_vecs = []
    anoms = list(read_test_dataset(dataset_dir + 'anom-test.jsonl'))
    for anom in anoms:
        anom_test_vecs.append(model.infer_vector(anom['words']))

    x_test = norm_test_vecs + anom_test_vecs
    y_test = ['norm']*len(norm_test_vecs) + ['anom']*len(anom_test_vecs)

    y_pred = clf.predict(x_test)

    print('Accuracy  : {}'.format(accuracy_score(y_test, y_pred)))
    print('Precision : {}'.format(precision_score(y_test, y_pred, pos_label='anom')))
    print('Recall    : {}'.format(recall_score(y_test, y_pred, pos_label='anom')))