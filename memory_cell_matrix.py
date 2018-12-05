import tensorflow as tf
from utils import *

class MemoryMatrix:
    def __init__(self, basis, h_lengths, memory_length, memory_size, batch_size):
        self.memory = tf.Variable(tf.constant(1e-6, shape=(batch_size, memory_length, memory_size)))

    def read(self):
        return self.memory

    def write(self, w_write, write, erase):
        self.memory = self.memory * (1 - outer_prod(w_write, erase)) + outer_prod(w_write, write)
        return 0
