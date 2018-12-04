import numpy as np


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def cos_sim_norm(v1, v2):
    return np.dot(v1, v2)


if __name__ == '__main__':
    x1 = np.array([1, 2, 3, 4])
    x2 = np.array([1, 2, 6, 5])

    sim1 = cos_sim(x1, x2)
    sim2 = cos_sim(x1 / np.linalg.norm(x1), x2 / np.linalg.norm(x2))

    print(sim1)
    print(sim2)