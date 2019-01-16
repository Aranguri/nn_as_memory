from PyDictionary import PyDictionary
from pearson_dict import pearson_meaning
import pickle
import pandas as pd
import numpy as np
from util import *

EMBEDDINGS_PATH = '../../nns/datasets/glove/glove.6B.50d.pickle'
LOCAL_WORDS = '/usr/share/dict/words'
WIKI_WORDS = 'wiki-100k.notatxt'

class DictTask:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.dict1 = PyDictionary().meaning
        self.dict2 = pearson_meaning
        with open(EMBEDDINGS_PATH, 'rb') as handle:
            self.weights = pickle.load(handle)
        self.words_list = [x.strip().lower() for x in open(LOCAL_WORDS, 'r')]

    def load_from_file(self, file):
        with open(file, 'rb') as handle:
            dataset = pickle.load(handle)
        ps(dataset)
        # for case in dataset:

    def embed(self, word):
        if word not in self.weights.keys():
            dims = len(self.weights['a'])
            self.weights[word] = np.random.randn(dims, stddev=.7)
        return self.weights[word]

    def embed_sentence(self, sentence):
        return np.array([self.embed(w) for w in sentence.split()])

    def next_case(self):
        correct = False

        while not correct:
            word = np.random.choice(self.words_list)
            m1, m2 = self.dict1(word), self.dict2(word)
            if (m1 and m2) != None and m1 != []:
                first_key = list(m1.keys())[0]
                if len(m1[first_key]) != 0:
                    correct = True

        m1 = m1[first_key][0]
        m1_embedded = self.embed_sentence(m1)
        m2_embedded = self.embed_sentence(m2)
        return m1_embedded, m2_embedded, self.embed(word), m1, m2, word

    def next_batch(self):
        case = list(zip(*[self.next_case() for i in range(self.batch_size)]))
        x1, x2, y, label = np.array(case[0]), np.array(case[1]), np.array(case[2]), case[3:6]
        max_length1 = np.max([v.shape[0] for v in x1])
        max_length2 = np.max([v.shape[0] for v in x2])
        max_length = np.maximum(max_length1, max_length2)

        def pad(x):
            for i in range(x.shape[0]):
                padding = ((0, max_length - x[i].shape[0]), (0, 0))
                x[i] = np.pad(x[i], padding, 'constant')
            x = np.array([np.array(v) for v in x])
            return x

        return pad(x1), pad(x2), y, label

    def nearest_words(self, vector):
        dists = []
        for other_word in self.words_list:
            other_vector = self.embed(other_word)
            dists.append((cosine_distance(vector, other_vector), other_word))
        return sorted(dists)[:10]

class LoadedDictTask:
    def __init__(self, batch_size, num_batches):
        with open(f'data/tasks_150_{batch_size}_{num_batches}_v2.pickle', 'rb') as handle:
            self.data = pickle.load(handle)
        self.num_batches = num_batches - 2
        self.i = 0

    def next_batch(self):
        self.i = self.i + 1 if self.i < self.num_batches else 0
        return self.data[self.i]

    def dev_batch(self):
        return self.data[-1]

def get_words_and_defs():
    data, dict1, dict2 = [], PyDictionary().meaning, pearson_meaning
    words = [x.strip().lower() for x in open(WIKI_WORDS, 'r')][1:]

    for i, word in enumerate(words):
        m1 = dict1(word)
        m2 = dict2(word)
        if (m1 and m2) != None and m1 != []:
            first_key = list(m1.keys())[0]
            if len(m1[first_key]) != 0:
                # print(word)
                data.append((m1, m2, word))

        if i % 200 == 0:
            print (i)
            with open(f'data/words_and_defs_{i}.pickle', 'wb') as handle:
                pickle.dump(data, handle)

get_words_and_defs()

def get_case():
    pass

def get_batch():
    case = list(zip(*[self.next_case() for i in range(self.batch_size)]))
    x1, x2, y, label = np.array(case[0]), np.array(case[1]), np.array(case[2]), case[3:6]
    max_length1 = np.max([v.shape[0] for v in x1])
    max_length2 = np.max([v.shape[0] for v in x2])
    max_length = np.maximum(max_length1, max_length2)

    def pad(x):
        for i in range(x.shape[0]):
            padding = ((0, max_length - x[i].shape[0]), (0, 0))
            x[i] = np.pad(x[i], padding, 'constant')
        x = np.array([np.array(v) for v in x])
        return x

    return pad(x1), pad(x2), y, label

def get_dataset():
    pass

def store_tasks(num_batches, batch_size, start=0):
    dict_task = DictTask(batch_size)
    data = []
    if start != 0:
        with open(f'data/tasks_{num_batches}_{batch_size}_{start}.pickle', 'rb') as handle:
            data = pickle.load(handle)
    for i in range(start, num_batches):
        print(f'{i}/{num_batches}')
        data.append(dict_task.next_batch())
        with open(f'data/tasks_{num_batches}_{batch_size}_{i}.pickle', 'wb') as handle:
            pickle.dump(data, handle)


'''
from task import DictTask
dict_task = DictTask(16)
a, b, c = dict_task.next_batch()
a.shape
'''
