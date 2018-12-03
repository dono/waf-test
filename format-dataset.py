import re
import random
import gzip
import json
import urllib.parse


reg = re.compile(r'(=|&|<|>|\(|\)|\.|/| |--|\r\n|\n\r|\n|\r)')
def split_payload(str):
    str = str.lower() # 大文字小文字を区別しない
    return reg.split(str)

def decode_payload(str):
    # double decoding
    str = urllib.parse.unquote_plus(str)
    str = urllib.parse.unquote_plus(str)
    return str

def parse_dataset(filepath):
    text = ''
    with gzip.open(filepath, mode='rt') as f:
        for line in f:
            if ('GET' in line) or ('POST' in line) or ('PUT' in line):
                if text != '':
                    yield text
                    text = ''
            text = text + line
        yield text

def parse_raw_http(str):
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

    # To be confirmed
    words = [method, u.netloc, *u.path.split(r'/')[1:], *split_payload(decode_payload(payload))]
    # words = split_payload(decode_payload(payload))

    return {'payload': payload, 'words': words}

def get_reqs(filepath, label):
    reqs = []
    for text in parse_dataset(filepath):
        req = parse_raw_http(text)
        req['label'] = label
        reqs.append(req)
    return reqs

if __name__ == '__main__':
    norm_train = get_reqs('./static/original/normalTrafficTraining.txt.gz', 'norm')
    norm_test = get_reqs('./static/original/normalTrafficTest.txt.gz', 'norm')
    anom_test = get_reqs('./static/original/anomalousTrafficTest.txt.gz', 'anom')

    # norm_train = [req for req in norm_train if req['payload'] != '']
    # norm_test = [req for req in norm_test if req['payload'] != '']
    # anom_test = [req for req in anom_test if req['payload'] != '']

    # print(len(norm_train))
    # print(len(norm_test))
    # print(len(anom_test))

    random.shuffle(norm_train)
    random.shuffle(norm_test)
    random.shuffle(anom_test)

    new_norm_train = norm_train[:5000]
    new_anom_train = anom_test[:5000]
    new_norm_test = norm_test[:1000]
    new_anom_test = anom_test[5000:6000]

    save_dir = './static/processed/v6/'

    with open(save_dir+'norm-train.jsonl', 'w') as f:
        for req in new_norm_train:
            f.write('{}\n'.format(json.dumps(req)))

    with open(save_dir+'anom-train.jsonl', 'w') as f:
        for req in new_anom_train:
            f.write('{}\n'.format(json.dumps(req)))
            
    with open(save_dir+'norm-test.jsonl', 'w') as f:
        for req in new_norm_test:
            f.write('{}\n'.format(json.dumps(req)))
            
    with open(save_dir+'anom-test.jsonl', 'w') as f:
        for req in new_anom_test:
            f.write('{}\n'.format(json.dumps(req)))
            