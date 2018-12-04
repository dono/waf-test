import ml
from distributed import Client

def gen_params():
    params = []
    for window in range(1, 6): # 5
        for min_count in range(0, 21): # 20
            for vector_size in range(10, 1001, 10): # 100
                for alpha in range(0.001, 0.11, 0.001): # 100
                    for epochs in range(10, 301, 10): # 30
                        params.append((window, min_count, vector_size, alpha, epochs))
    return params


if __name__ == '__main__':
    client = Client('localhost:8786')

    L = client.map(ml.get_score, params)
    for l in L:
        print(l.result())