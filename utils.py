import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def ps(a1, a2=None, a3=None, a4=None, a5=None):
    for a in [a1, a2, a3, a4, a5]:
        if a is not None:
            print (np.shape(a))

def tp(t1):
    return tf.Print([0], [ts(t1)])

def ts(a1):#todo: allow more than one array as input
    return tf.shape(a1)

def outer_prod(t1, t2):
    return tf.einsum('ij,ik->ijk', t1, t2)

def expand(t1, axis=0):
    return tf.expand_dims(t1, axis)

def plot(array):
    plt.ion()
    plt.cla()
    if type(array) is dict:
        array = [v for v in array.values()]
    xlim = 2 ** (1 + int(np.log2(len(array))))
    ylim = 2 ** (1 + int(np.log2(np.maximum(max(array), 1e-8))))

    plt.xlim(0, xlim)
    plt.ylim(0, ylim)#2000)#.6)
    plt.plot(array)
    plt.pause(1e-8)

ca = lambda t: tf.reduce_any(tf.is_nan(t))
cn = lambda t: tf.count_nonzero(t)
ap = lambda t: tf.reduce_any(t <= 0)

import numpy as np
import tensorflow as tf

# def expand(x, dim, N):
# return tf.concat([tf.expand_dims(x, dim) for _ in range(N)], axis=dim)

def learned_init(units):
    return tf.squeeze(tf.contrib.layers.fully_connected(tf.ones([1, 1]), units,
        activation_fn=None, biases_initializer=None))

def create_linear_initializer(input_size, dtype=tf.float32):
    stddev = 1.0 / np.sqrt(input_size)
    return tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
