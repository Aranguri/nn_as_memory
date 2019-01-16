from PyDictionary import PyDictionary
from pearson_dict import pearson_meaning
import pickle
import pandas as pd
import numpy as np
from util import *
from keras.preprocessing.sequence import pad_sequences

EMBEDDINGS_PATH = '../../nns/datasets/glove/glove.6B.50d.pickle'
LOCAL_WORDS = '/usr/share/dict/words'
WIKI_WORDS = 'wiki-100k.notatxt'

class DictTask:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.dict1 = PyDictionary().meaning
        self.dict2 = pearson_meaning
        # with open(EMBEDDINGS_PATH, 'rb') as handle:
            # self.weights = pickle.load(handle)
        # self.words_list = [x.strip().lower() for x in open(LOCAL_WORDS, 'r')]

    def load_from_file(self, file):
        with open(file, 'rb') as handle:
            dataset = pickle.load(handle)

        new_dataset, vocab, word_to_i = [], set([]), {}

        for m1, m2, w in dataset:
            # Get specific defs
            first_key = list(m1.keys())[0]
            m1 = m1[first_key][0]

            m2 = m2['senses'][0]['definition']
            m2 = m2 if type(m2) == str else m2[0]

            # Tokenize
            new_m1, new_m2 = [], []

            for word in m1.split():
                if word not in word_to_i.keys():
                    word_to_i[word] = len(word_to_i)
                new_m1.append(word_to_i[word])

            for word in m2.split():
                if word not in word_to_i.keys():
                    word_to_i[word] = len(word_to_i)
                new_m2.append(word_to_i[word])

            if w not in word_to_i.keys():
                word_to_i[w] = len(word_to_i)
            new_w = word_to_i[w]

            new_dataset.append((new_m1, new_m2, new_w, m1, m2, w))

        batches, i = [], 0

        while i + self.batch_size < len(new_dataset):
            batch = [[], [], [], [], [], []]

            for j in range(self.batch_size):
                new_m1, new_m2, new_w, m1, m2, w = new_dataset[i + j]
                batch[0].append(new_m1)
                batch[1].append(new_m2)
                batch[2].append(new_w)
                batch[3].append(m1)
                batch[4].append(m2)
                batch[5].append(w)

            i += j
            new_batch = [[], [], [], []]
            max_length1 = np.max([np.shape(v)[0] for v in batch[0]])
            max_length2 = np.max([np.shape(v)[0] for v in batch[1]])
            max_length = np.maximum(max_length1, max_length2)
            new_batch[0] = pad_sequences(batch[0], max_length, padding='post')
            new_batch[1] = pad_sequences(batch[1], max_length, padding='post')
            new_batch[2] = batch[2]
            new_batch[3] = batch[3:6]

            batches.append(new_batch)

        return batches[:-1], len(word_to_i)

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
    def __init__(self, batch_size):
        # with open(f'data/tasks_150_{batch_size}_{num_batches}_v2.pickle', 'rb') as handle:
            # self.data = pickle.load(handle)
        dict_task = DictTask(batch_size)
        self.data, self.vocab_size = dict_task.load_from_file('data/words_and_defs_5400.pickle')
        self.num_batches = len(self.data) - 2
        self.i = 0

    def next_batch(self):
        self.i = self.i + 1 if self.i < self.num_batches else 0
        return self.data[self.i]

    def dev_batch(self):
        return self.data[-1]

def get_words_and_defs():
    data, dict1, dict2 = [], PyDictionary().meaning, pearson_meaning
    words = [x.strip().lower() for x in open(WIKI_WORDS, 'r')][75000:]

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
            with open(f'data/words_and_defs_75000_{i}.pickle', 'wb') as handle:
                pickle.dump(data, handle)

get_words_and_defs()

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
