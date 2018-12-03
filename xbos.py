import json
import random
import warnings
import numpy as np
from concurrent import futures
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.cluster import KMeans
import numpy as np
from pandas import DataFrame
from math import pow
import math

class XBOS:
    def __init__(self,n_clusters=15,effectiveness=500,max_iter=2):
        self.n_clusters=n_clusters
        self.effectiveness=effectiveness
        self.max_iter=max_iter
        self.kmeans = {}
        self.cluster_score = {}
        
    def fit(self, data):
        length = len(data)
        for column in data.columns:
            kmeans = KMeans(n_clusters=self.n_clusters,max_iter=self.max_iter)
            self.kmeans[column]=kmeans
            kmeans.fit(data[column].values.reshape(-1,1))
            assign = DataFrame(kmeans.predict(data[column].values.reshape(-1,1)),columns=['cluster'])
            cluster_score=assign.groupby('cluster').apply(len).apply(lambda x:x/length)
            ratio=cluster_score.copy()
        
            sorted_centers = sorted(kmeans.cluster_centers_)
            max_distance = ( sorted_centers[-1] - sorted_centers[0] )[ 0 ]
        
            for i in range(self.n_clusters):
                for k in range(self.n_clusters):
                    if i != k:
                        dist = abs(kmeans.cluster_centers_[i] - kmeans.cluster_centers_[k])/max_distance
                        effect = ratio[k]*(1/pow(self.effectiveness,dist))
                        cluster_score[i] = cluster_score[i]+effect
                        
            self.cluster_score[column] = cluster_score
                    
    def predict(self, data):
        length = len(data)
        score_array = np.zeros(length)
        for column in data.columns:
            kmeans = self.kmeans[ column ]
            cluster_score = self.cluster_score[ column ]
            
            assign = kmeans.predict( data[ column ].values.reshape(-1,1) )
            #print(assign)
            
            for i in range(length):
                score_array[i] = score_array[i] + math.log10( cluster_score[assign[i]] )
            
        return score_array
    
    def fit_predict(self,data):
        self.fit(data)
        return self.predict(data)

# jsonl形式のトレーニング用データセットを読み込む
def read_train_dataset(path):
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            # if len(d['words']) != 0: # ペイロードが空のデータは除外する
            yield TaggedDocument(d['words'], [d['label']])

# jsonl形式のテスト用データセットを読み込む
def read_test_dataset(path):
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            # if len(d['words']) != 0: # ペイロードが空のデータは除外する
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


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=FutureWarning)

    # 正常データだけ学習
    train = list(read_train_dataset('./norm-train.jsonl'))
    model = Doc2Vec(train, dm=1, window=2, min_count=1, vector_size=300, alpha=0.08, min_alpha=0.01, epochs=600, workers=6)

    norm_vecs = []
    anom_vecs = []

    for d in norm_test:
        vec = model.infer_vector(d['words'])
        norm_vecs.append(vec)

    for d in anom_test:
        vec = model.infer_vector(d['words'])
        anom_vecs.append(vec)

