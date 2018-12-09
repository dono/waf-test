import re
import gzip
import imp
import urllib.parse
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


ascii_symbols = r'[ -/:-@[-`{-~]' # asciiコード中の記号(スペース含む)とマッチする正規表現
ascii_control_chars = r'[\x00-\x1f]|\x7f' # asciiコード中の制御文字とマッチする正規表現
reg = re.compile('({}|{})'.format(ascii_symbols, ascii_control_chars))
def split_text(str):
    str = str.lower() # 大文字をすべて小文字に変換
    return reg.split(str)

# データセットから1件ずつHTTPテキストをパースして取得
def dataset2texts(filepath):
    text = ''
    with gzip.open(filepath, mode='rt') as f:
        for line in f:
            if ('GET' in line) or ('POST' in line) or ('PUT' in line):
                if text != '':
                    yield text
                    text = ''
            text = text + line
        yield text

# HTTPテキストを単語配列に変換
def text2words(str):
    arr = str.split('\n')
    method, url, _ = arr[0].split(' ')
    u = urllib.parse.urlparse(url)

    payload = ''
    if method == 'GET':
        payload = u.query
    elif method == 'POST' or method == 'PUT':
        for line in reversed(arr):
            if line == '':
                continue
            else:
                payload = line
                break

    payload = urllib.parse.unquote_plus(payload) # decoding

    words = split_text(u.path) + split_text(payload)
    return words


if __name__ == '__main__':
    norm_train = list(dataset2texts('./static/original/normalTrafficTraining.txt.gz'))
    anom_test = list(dataset2texts('./static/original/anomalousTrafficTest.txt.gz'))

    X = np.array(norm_train + anom_test)
    y = np.array(['norm'] * len(norm_train) + ['anom'] * len(anom_test))

    kf = KFold(n_splits=10, shuffle=True, random_state=0)

    result_sum = {'Accuracy': 0, 'Precision': 0, 'Recall': 0}
    for train_idx, test_idx in kf.split(X):
        y_train, y_test = y[train_idx], y[test_idx]

        # BoWベクトライザの生成
        vectorizer = CountVectorizer(analyzer=text2words)
        vectorizer.fit(X[train_idx])

        # ベクトル生成
        X_train = vectorizer.transform(X[train_idx])
        X_test = vectorizer.transform(X[test_idx])

        # ランダムフォレストでテスト
        clf = RandomForestClassifier(random_state=0, n_estimators=10, n_jobs=6)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        Accuracy = accuracy_score(y_test, y_pred)
        Precision = precision_score(y_test, y_pred, pos_label='norm')
        Recall = recall_score(y_test, y_pred, pos_label='norm')

        result_sum['Accuracy'] += Accuracy
        result_sum['Precision'] += Precision
        result_sum['Recall'] += Recall

    print('Accuracy:  {0:.4f}'.format(Accuracy))
    print('Precision: {0:.4f}'.format(Precision))
    print('Recall:    {0:.4f}'.format(Recall))