import pickle
import numpy as np

with open(f'data/tasks_150_32_140.pickle', 'rb') as handle:
    data = pickle.load(handle)

new_data = []

for i in range(len(data)):
    max_length = np.maximum(len(data[i][0][0]), len(data[i][1][0]))
    def pad(x):
        new_x = []
        for i in range(x.shape[0]):
            padding = ((0, max_length - x[i].shape[0]), (0, 0))
            new_x.append(np.pad(x[i], padding, 'constant'))
        new_x = np.array([np.array(v) for v in new_x])
        return new_x
    defs1 = pad(data[i][0])
    defs2 = pad(data[i][1])
    case = defs1, defs2, data[i][2], data[i][3]
    new_data.append(case)

with open(f'data/tasks_150_32_140_v2.pickle', 'wb') as handle:
    pickle.dump(new_data, handle)
