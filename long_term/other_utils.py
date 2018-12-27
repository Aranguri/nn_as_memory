import numpy as np
import matplotlib.pyplot as plt

def cosine_distance(v1, v2):
    return np.linalg.norm(v1 - v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def plot(array):
    plt.ion()
    plt.cla()
    if type(array) is dict:
        array = [v for v in array.values()]
    xlim = 2 ** (1 + int(np.log2(len(array))))
    ylim = 2 ** (1 + int(np.log2(np.maximum(max(array), 1e-8))))

    plt.xlim(0, xlim)
    plt.ylim(0, ylim)
    plt.plot(array)
    plt.pause(1e-8)

def ps(array):
    print(np.shape(array))
