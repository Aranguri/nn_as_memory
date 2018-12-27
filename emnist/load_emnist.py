import sys
sys.path.append('../')
import numpy as np
from mlxtend.data import loadlocal_mnist
from utils import *

'''
xs, ys = loadlocal_mnist(
        images_path='/home/aranguri/Desktop/dev/nns/datasets/gzip/emnist-byclass-train-images-idx3-ubyte',
        labels_path='/home/aranguri/Desktop/dev/nns/datasets/gzip/emnist-byclass-train-labels-idx1-ubyte')

np.save('emnist_x', xs[:6000])
np.save('emnist_y', ys[:6000])
'''

xs = np.load('emnist_x.npy')
ys = np.load('emnist_y.npy')

xs_by_class = [xs[ys == y] for y in set(ys)]
train_size = min([x.shape[0] for x in xs_by_class])

xs = xs_by_class[:, :train_size]
