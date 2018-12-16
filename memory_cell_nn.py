import itertools
import time
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from utils import *
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist

class MemoryNN:
    def __init__(self, basis, h_lengths, memory_length, memory_size, batch_size):
        self.ws, self.bs, self.xs = {}, {}, {0: basis}
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        lengths = h_lengths + [memory_length]

        for i, (m, n) in enumerate(zip(lengths, lengths[1:])):
            self.ws[i] = tf.get_variable(f'w{i}', (m, n))
            self.bs[i] = tf.get_variable(f'b{i}', (n, memory_size))

    def read(self):
        for i, (w, b) in enumerate(zip(self.ws.values(), self.bs.values())):
            self.xs[i+1] = tf.nn.leaky_relu(tf.einsum('ijk,jl->ilk', self.xs[i], w) + b)
        return self.xs[len(self.ws)]

    def write(self, w_write, write, erase):
        #there could be no self.xs
        memory = self.xs[len(self.ws)] * (1 - outer_prod(w_write, erase)) + outer_prod(w_write, write)
        loss = tf.losses.mean_squared_error(memory, self.xs[len(self.ws)])
        grads = tf.gradients([loss], [self.ws[0], self.bs[0]])
        minimize = self.optimizer.apply_gradients(zip(grads, [self.ws[0], self.bs[0]]))
        return loss
