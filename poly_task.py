import numpy as np

class PolyTask:
    def __init__(self, batch_size, seq_length):
        self.batch_size = batch_size
        self.seq_length = seq_length

    def next_batch(self):
        a, b, c = np.random.randn(3, self.batch_size, 1)
        xs = np.random.randn(self.batch_size, self.seq_length - 1)
        ys = a * xs ** 2 + b * xs + c#a * xs ** 2 + b * xs + c
        xs = np.pad(xs, ((0, 0), (0, 1)), 'constant')
        ys_start_pad = np.pad(ys, ((0, 0), (1, 0)), 'constant')
        ys_end_pad = np.pad(ys, ((0, 0), (0, 1)), 'constant')
        mixed = list(zip(ys_start_pad.flatten(), xs.flatten()))
        xs = np.reshape(mixed, (self.batch_size, self.seq_length, 2))
        ys = np.expand_dims(ys_end_pad, 2)
        return xs, ys
