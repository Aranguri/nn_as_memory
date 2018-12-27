import itertools
import time
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist
import sys
sys.path.append('../')
from utils import *
from memory_cell_nn import MemoryNN

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784)
x_train = np.array([x_train[y_train == i][:5000] for i in range(10)])
xs = np.swapaxes(x_train, 0, 1)
'''
xs, ys = loadlocal_mnist(
    images_path='/home/aranguri/Desktop/dev/nns/datasets/gzip/emnist-byclass-train-images-idx3-ubyte',
    labels_path='/home/aranguri/Desktop/dev/nns/datasets/gzip/emnist-byclass-train-labels-idx1-ubyte')
'''
xs = np.load('../data/emnist_x.npy')
ys = np.load('../data/emnist_y.npy')

xs_by_class = np.array([xs[ys == y] for y in set(ys)])
train_size = min([x.shape[0] for x in xs_by_class])
xs = np.array([x[:train_size] for x in xs_by_class])
xs = np.swapaxes(xs, 0, 1)

'''
next steps:
* add bias: does it need to have size (h_length x h_size) or only (h_length,)
* test whether this works with the example below
* train the memroy cell to remember mnist digits. look at the cost and visualize the output of the memorycell
* run ntm.py three or four times to get a flavour of the range of values it reaches. (aka benchmark)
* integrate this mem module to the net.py
* flavour. (aka benchmark)

mnist thing:
* what are we doin' ?
* use l1 regularization (ie use only the sign.)

what does the batch_size mean?

it's interesting, the model learns to create matter and anti matter
'''

batch_size = 1
input_length = 8
h_lengths = [input_length, 20, 40]
memory_length = xs.shape[1]
memory_size = 784
reg_bias = tf.constant(30, dtype=tf.float32)

basis = tf.get_variable('basis', [input_length, memory_size])
ys = tf.placeholder(tf.float32, [memory_length, memory_size])
memory = MemoryNN(basis, h_lengths, memory_length, memory_size, batch_size)
output = memory.read()

optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
loss = tf.losses.mean_squared_error(output, ys)
l1 = tf.contrib.layers.l1_regularizer(reg_bias)
loss += l1(memory.bs[0]) + l1(memory.bs[1]) + l1(memory.bs[2])
amount = tf.reduce_sum(tf.to_float(memory.bs[2] > 1e-3))
minimize = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tr_loss = {}
    fig = plt.figure()
    for l in itertools.count():
        for i, ys_ in enumerate(xs):
            step = i + l * len(xs)
            _, tr_loss[step], output_, bs_, xs_, amount_ = sess.run([minimize, loss, output, memory.bs, memory.xs, amount], feed_dict={ys: ys_})

            if step % 10 == 0:
                print(f'Step: {step}. loss: {tr_loss[step]}')
                print(bs_[0])
                print(amount_)

            if step % 250 == 0:
                plt.ion()
                plt.cla()
                for j, v in enumerate(list(xs_.values())):
                    fig.add_subplot(4, 1, 4 - j)
                    plt.imshow(v.reshape(-1, 28).T)

                plt.pause(1e-8)

'''
It's interesting: the nn can't overfit to the trainig data. We could have something like regularization going on here. or maybe it's not working at all :)
we could penlize the loss in a certain way so as to make memories similar to one specific type of the class. for instance, if we want to remember ones, we penalize the model in a way that prefers being near a subset of the examples instead of being not-that-near wrt to a bigger subset of the examples.
it would be good to allow one transformation.

I don't know how to enforce sparsity in the biases. we want a real

emnist. lr 3e-2
10-20-40: visual test passed. not perfect characters though.
5-20-40: vt passed at 550 steps (1 training ex for every char.)
5-20-50: vt passed at 1200 steps (15 training ex)
2-20-50: at 8000
1-20-50: doesn't pass the vt at 35000



'''
