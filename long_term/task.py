from PyDictionary import PyDictionary
import pickle
import pandas as pd
import numpy as np

EMBEDDINGS_PATH = '../../nns/datasets/glove/glove.6B.50d.pickle'
LOCAL_WORDS = '/usr/share/dict/words'

class DictTask:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.dict = PyDictionary()
        with open(EMBEDDINGS_PATH, 'rb') as handle:
            self.weights = pickle.load(handle)
        self.words_list = [x.strip().lower() for x in open(LOCAL_WORDS, 'r')]

    def embed(self, word):
        if word not in self.weights.keys():
            dims = len(self.weights['a'])
            self.weights[word] = np.random.randn(dims)#TODo: add stddev that is the same as that of glove
        return self.weights[word]

    def embed_sentence(self, sentence):
        return np.array([self.embed(w) for w in sentence.split()])

    def next_case(self):
        correct = False
        while not correct:
            word = np.random.choice(self.words_list)
            meaning = self.dict.meaning(word)
            if meaning == None or meaning == []:
                pass
            elif len(meaning[list(meaning.keys())[0]]) == 0:
                pass
            else:
                correct = True

        print(list(meaning.keys()))
        meaning = meaning[list(meaning.keys())[0]][0]
        return self.embed_sentence(meaning), self.embed(word), meaning, word

    def next_batch(self):
        case = list(zip(*[self.next_case() for i in range(self.batch_size)]))
        x, y, label = np.array(case[0]), np.array(case[1]), case[2:4]
        max_length = np.max([v.shape[0] for v in x])

        for i in range(x.shape[0]):
            padding = ((0, max_length - x[i].shape[0]), (0, 0))
            x[i] = np.pad(x[i], padding, 'constant')

        x = np.array([np.array(v) for v in x])

        return x, y, label

def store_tasks(num_batches, batch_size):
    dict_task = DictTask(batch_size)
    data = [dict_task.next_batch() for i in range(num_batches)]
    with open(f'data/generated_tasks_{num_batches}_{batch_size}.pickle', 'wb') as handle:
        pickle.dump(data, handle)


'''
from task import DictTask
dict_task = DictTask(16)
a, b, c = dict_task.next_batch()
a.shape
'''
