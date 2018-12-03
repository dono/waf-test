from spherecluster import SphericalKMeans
import numpy as np
from pandas import DataFrame
from xbos import XBOS


if __name__ == '__main__':
    # xbos = XBOS(n_clusters=5)
    # norm_train = list(read_train_dataset('./static/norm-train.jsonl'))
    # anom_train = list(read_train_dataset('./static/anom-train.jsonl'))

    # model = Doc2Vec([*norm_train, *anom_train], dm=1, window=2, min_count=13, vector_size=200, alpha=0.08, min_alpha=0.01, epochs=160, workers=6)

    # norm_train_vecs = [model.docvecs['norm'+str(i)] for i in range(len(norm_train))]
    # anom_train_vecs = [model.docvecs['anom'+str(i)] for i in range(len(anom_train))]

    # X = norm_train_vecs + anom_train_vecs

    # skm = SphericalKMeans(n_clusters=15, init='k-means++')
    # skm.fit(X)
    data = DataFrame(data={'attr1':[1,1,1,1,2,2,2,2,2,2,2,2,3,5,5,6,6,7,7,7,7,7,7,7,15],'attr2':[1,1,1,1,2,2,2,2,2,2,2,2,3,5,5,6,6,7,7,7,13,13,13,14,15]})
    xbos = XBOS(n_clusters=3)
    xbos.fit(data)
    result = xbos.predict(DataFrame([1]))
    print(result)
