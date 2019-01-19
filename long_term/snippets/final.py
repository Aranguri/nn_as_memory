from nltk.corpus import wordnet as wn
import pickle

with open('all.json', 'rb') as handle:
    dict1 = pickle.load(handle)

data, counter = [], 0

for word, definition in list(dict1.items())[30:]:
    pos, text = definition[0].values()
    def1 = f'{pos} {text}'

    if len(wn.synsets(word)) > 0:
        def2 = wn.synsets(word)[0].definition()
        data.append((def1, def2, word))

with open('../54k.pickle', 'wb') as handle:
    pickle.dump(data, handle)
