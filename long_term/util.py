# Some packages are here to be automatically imported when we import this file
import matplotlib.pyplot as plt
import pickle
import itertools
import numpy as np
import tensorflow as tf
# import colorful
import re

relu = lambda x: np.maximum(0, x)
relu_prime = lambda x: (x > 0)# * x
sigmoid = lambda x: 1 / (1 + np.exp(-x))
sigmoid_prime = lambda x: (1 - sigmoid(x)) * sigmoid(x)
tanh = lambda x: np.tanh(x)
tanh_prime = lambda x: 1 - np.square(np.tanh(x))
identity = lambda x: x
identity_prime = lambda x: 1

def psf(t):
    return tf.matmul(t, tf.constant(0.0))

def psi(t):
    return tf.matmul(t, tf.to_int32(tf.constant(0)))

def softmax(values):
    exp = np.exp(values)
    return exp / np.sum(exp)

def expand(array):
    return np.expand_dims(array, axis=1)

def enlarged(array, times):
    add = np.random.randn(array.shape[0], times) * 1e-3
    return np.concatenate((array, add), 1)

def clean(a):
    a = a - a.mean()
    a = a / a.std()
    return a

def add_bias(array, axis=0):
    pad = ((0, 1), (0, 0)) if axis == 0 else ((0, 0), (0, 1))
    return np.pad(array, pad, 'constant', constant_values=1)

def remove_bias(array, axis=0):
    return array[:-1] if axis == 0 else array[:, :-1]

def ps(a1, a2=None, a3=None, a4=None, a5=None):
    for a in [a1, a2, a3, a4, a5]:
        if a is not None:
            print (np.shape(a))

def normalize(array):
    array = array[:, 0]
    array = array - np.min(array)
    return array / np.sum(array)

def one_hot(pos, length, embed=False):
    if embed:
        array = np.zeros((length * len(pos)))
        for i, p in enumerate(pos):
            array[i * length + p] = 1
    elif type(pos) == list or type(pos) == np.ndarray:
        array = np.zeros((len(pos), length))
        for i, p in enumerate(pos):
            array[i][p] = 1
    else:
        array = np.zeros((length))
        array[pos] = 1
    return array

def points_to_curve(y):
    x = range(len(y))
    curve = np.poly1d(np.polyfit(x, y, 8))(np.unique(x))
    return curve

def eval_numerical_gradient(f, x, dout=None, h=1e-4):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])#, op_flags=['readwrite'])

    while not it.finished:
        i = it.multi_index

        old_xi = x[i]
        x[i] = old_xi + h
        pos = f(x)[0]
        x[i] = old_xi - h
        neg = f(x)[0]
        x[i] = old_xi

        if type(dout) != type(None):
            grad[i] = np.sum((pos - neg) * dout) / (2 * h)
        else:
            grad[i] = np.sum((pos - neg)) / (2 * h)

        it.iternext()
    return grad

def rel_difference(a1, a2):
    print (a1)
    print (a2)
    den = np.maximum.reduce([np.abs(a1), np.abs(a2), np.ones_like(a2) * 1e-15])
    print (np.max(np.abs(a1 - a2) / den))

def dict_max(d):
    return np.max(list(d.values()))

def dict_mean(d, start=0):
    return np.mean(list(d.values())[start:])

def dict_sum(d):
    values = np.array([v for v in d.values()])
    return values.sum(0)

def rec_sum(array):
    if type(array) == list or len(array.shape) == 1:
        return np.sum([rec_sum(a) for a in array])
    else:
        return np.sum(array)

def shuffle(array, axis):
    array = array.swapaxes(0, axis)
    np.random.shuffle(array)
    array = array.swapaxes(axis, 0)
    return array

def save(name, *args):
    pass#with open(f'savings/{name}.pkl', 'wb') as f:
    #    pickle.dump(args, f)

def restore(name):
    pass#with open(f'savings/{name}.pkl', 'rb') as f:
    #    return pickle.load(f)

def init_cprint():
    colors = {'black': (0, 0, 0)}
    for i in range(0, 256):
        colors[str(i)] = (i, 255, i)
        colors[str(i + 256)] = (255, 255-i, 255-i)
        colorful.use_palette(colors)
        colorful.update_palette(colors)

def cprint(char, color):
    color = (abs(color) ** (1/3)) * (color / abs(color)) + 1
    color = str(int(color * 255))
    #print (getattr(colorful, f'black_on_{color}')(char), end='')

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

def smooth_plot(values, smooth_factor=10):
    plot([np.mean(list(values.values())[0:(i + 1)]) if i < (smooth_factor + 1) else np.mean(list(values.values())[i-smooth_factor:(i + 1)]) for i in range(len(values))])

def multiremove(text, items, regex=False):
    for item in items:
        if regex:
            text = re.sub(item, '', text)
        else:
            text = text.replace(item, '')
    return text

def clean_text(text):
    left = ["'", ')', '.', ',', ':', ';', '?', '!']
    right = ['(']
    both = ['[', ']', '-', '—', '"', '\n']

    text = text.lower()

    for c in left:
        text = text.replace(c, ' ' + c)

    for c in right:
        text = text.replace(c, c + ' ')

    for c in both:
        text = text.replace(c, ' ' + c + ' ')

    #TODO: add spaces between numbers. eg 1954 => 1 9 5 4
    text = text.replace('  ', ' ')
    words = text.split(' ')
    # words = [w for w in words if w.isalpha()]
    return words

def tokenize_words_simple(words):
    unique_words = set(words)
    vocab_size = len(unique_words)
    word_to_i = {w: i for i, w in enumerate(unique_words)}
    i_to_word = np.array([w for w in unique_words])
    x = np.array([word_to_i[w] for w in words])
    return vocab_size, word_to_i, i_to_word, x

def tokenize_words(words, word_to_i={}):
    x, new_words = [], []
    for word in words:
        if word not in word_to_i.keys():
            word_to_i[word] = len(word_to_i)
            new_words.append(word)
        x.append(word_to_i[word])
    return x, word_to_i, new_words

def conv_structure(inpt, layers, verbose=False):
    if verbose: print (inpt)
    weight = [[]] * len(layers)
    for i, layer in enumerate(layers):
        weight[i] = layer[0], inpt[1], layer[1]
        inpt[0] = inpt[0] - layer[0] + 1
        inpt[1] = layer[1]
        if verbose: print (inpt)
    return weight, inpt[0] * inpt[1]

#todo: limit only three digits
def debug(i, tr, dev, debug_steps, plot='none'):
    tr, dev = list(tr.values()), list(dev.values())

    print(f'{i}) Train: {np.mean(tr[-10:])} Dev: {np.mean(dev[-10:])}')

    if plot == 'tr' or plot == 'both': smooth_plot(tr, 300)
    if plot == 'dev' or plot == 'both': smooth_plot(dev, 300)
