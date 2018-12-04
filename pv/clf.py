import json
import random
import warnings
import numpy as np
from concurrent import futures
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


# jsonl形式のトレーニング用データセットを読み込む
def read_train_dataset(path):
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            yield TaggedDocument(d['words'], [d['label']])

# jsonl形式のテスト用データセットを読み込む
def read_test_dataset(path):
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            yield d

def norm_test(model, norm_test_data):
    TP = FN = 0
    for data in norm_test_data:
        vec = model.infer_vector(data['words'])
        label = model.docvecs.most_similar([vec], topn=1)[0][0]
        if label == 'norm':
            TP += 1
        else:
            FN += 1
    return TP, FN

def anom_test(model, anom_test_data):
    FP = TN = 0
    for data in anom_test_data:
        vec = model.infer_vector(data['words'])
        label = model.docvecs.most_similar([vec], topn=1)[0][0]
        if label == 'norm':
            FP += 1
        else:
            TN += 1
    return FP, TN


def get_score(window, min_count, vector_size, alpha, min_alpha, epochs, dataset_dir):
    warnings.filterwarnings('ignore', category=FutureWarning)


    train = list(read_train_dataset(dataset_dir+'norm-train.jsonl')) + list(read_train_dataset(dataset_dir+'anom-train.jsonl'))
    model = Doc2Vec(train, dm=1, window=window, min_count=min_count, vector_size=vector_size, alpha=alpha, min_alpha=min_alpha, epochs=epochs, workers=6)

    norms = list(read_test_dataset(dataset_dir+'norm-test.jsonl'))
    anoms = list(read_test_dataset(dataset_dir+'anom-test.jsonl'))

    with futures.ProcessPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(norm_test, model, norms)
        f2 = executor.submit(anom_test, model, anoms)
        TP, FN = f1.result()
        FP, TN = f2.result()

        print('TP: {}, FN: {}, FP: {}, TN: {}'.format(TP, FN, FP, TN))

        Accuracy = 0
        Precision = 0
        Recall = 0
        F1 = 0

        if (TP + FP) != 0 and (TP + FN) != 0 and (TP + FP + FN + TN) != 0:
            Accuracy = (TP + TN) / (TP + FP + FN + TN)
            Precision = TP / (TP + FP)
            Recall = TP / (TP + FN)
            F1 = 2 * Recall * Precision / (Recall + Precision)

        result = {'params': {'window': window, 'min_count': min_count, 'vector_size': vector_size, 'alpha': alpha, 'min_alpha': min_alpha, 'epochs': epochs},
                  'score': {'Accuracy': Accuracy, 'Precision': Precision, 'Recall': Recall, 'F1': F1}}
        return result


if __name__ == '__main__':
    # result = get_score(2, 1, 300, 0.08, 0.01, 400)
    # for epoch in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    dataset_dir = './static/processed/v6/'
    result = get_score(2, 1, 600, 0.08, 0.01, 300, dataset_dir)
    print(json.dumps(result))
