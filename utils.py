import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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
    plt.ylim(0, ylim)
    plt.plot(array)
    plt.pause(1e-8)

# debugging tools
def ps(a1, a2=None, a3=None, a4=None, a5=None):
    for a in [a1, a2, a3, a4, a5]:
        if a is not None:
            print (np.shape(a))

tp = lambda t: tf.Print([0], [ts(t)])
ts = lambda t: tf.shape(t)#todo: allow more than one array as input
ca = lambda t: tf.reduce_any(tf.is_nan(t))
cn = lambda t: tf.count_nonzero(t)
ap = lambda t: tf.reduce_any(t <= 0)
