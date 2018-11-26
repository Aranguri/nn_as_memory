import tensorflow as tf
import collections
from utils import *
from lstm_layer import LSTM

NTMState = collections.namedtuple('NTMState', ('ctrl_state', 'read_list', 'w_list', 'memory'))

class NTMCell(tf.contrib.rnn.RNNCell):
    def __init__(self, output_size, batch_size, h_size, memory_length, memory_size, shift_length):
        interface_size = memory_size + 1 + 1 + shift_length + 1
        ctrl_size = 2 * interface_size + 2 * memory_size + output_size
        self.controller = LSTM(tf.zeros([2, batch_size, h_size]), ctrl_size)
        self.sizes = [batch_size, memory_length, memory_size, shift_length, output_size, interface_size]

    def __call__(self, x, prev_state):
        #prepare input
        batch_size, memory_length, memory_size, _, output_size, interface_size = self.sizes
        self.controller.h, read_list, w_prev, memory = prev_state
        x_and_r = tf.squeeze(tf.concat(([x], read_list), axis=2))

        #execute controller
        interface = tf.split(self.controller(x_and_r), [interface_size, interface_size, memory_size, memory_size, output_size], axis=1)
        interface_read, interface_write, write, erase, output = interface

        #read head
        w_read = self.addressing(interface_read, memory, w_prev[0])
        read_list = expand(tf.einsum('ij,ijk->ik', w_read, memory))

        #write head
        write, erase = tf.tanh(write), tf.sigmoid(erase)
        w_write = self.addressing(interface_write, memory, w_prev[1])
        memory = memory * (1 - outer_prod(w_write, erase)) + outer_prod(w_write, write)

        # prepare output
        # print_op = tf.Print([0], [memory[0][0]])
        # with tf.control_dependencies([print_op]):
        w_list = tf.concat((w_read, w_write), axis=0)
        return output, NTMState(ctrl_state=self.controller.h, read_list=read_list, w_list=w_list, memory=memory)

    def addressing(self, interface, memory, w_prev):
        # prepare input
        batch_size, memory_length, memory_size, shift_length = self.sizes[:4]
        key, gate, b, shift, sharpener = tf.split(interface, [memory_size, 1, 1, shift_length, 1], axis=1)
        key, gate, b = tf.tanh(key), tf.sigmoid(gate), tf.nn.softplus(b)
        shift, sharpener = tf.nn.softmax(shift, axis=1), (tf.nn.softplus(sharpener) + 1)
        shift = tf.pad(shift, tf.constant([[0, 0,], [0, memory_length - shift_length]]))

        # gate between content-based weight and previous weight
        unnorm_similarity = tf.einsum('ik,ijk->ij', key, memory)
        similarity = unnorm_similarity / (tf.norm(memory, axis=2) * tf.norm(key, axis=1, keepdims=True) + 1e-8)
        w_c = tf.nn.softmax(b * similarity, axis=1)
        w_g = gate * w_c + (1 - gate) * w_prev

        # convolve
        roll = lambda i: tf.manip.roll(shift, shift=((i-1) % memory_length), axis=1)
        rolled_matrix = tf.map_fn(roll, tf.range(memory_length), dtype=tf.float32)
        w_tilde = tf.einsum('ijk,jk->ji', rolled_matrix, w_g)
        w_tilde_num = tf.pow(w_tilde, sharpener)
        w = w_tilde_num / tf.reduce_sum(w_tilde_num, axis=1, keepdims=True)

        return w

    def zero_state(self, batch_size, dtype):
        batch_size, memory_length, memory_size = self.sizes[:3]
        #try self.controller.zero_state()
        read_list = tf.Variable(tf.constant(0.0, shape=(1, batch_size, memory_size)))
        w_list = tf.random_normal(shape=(2 * batch_size, memory_length), stddev=1e-5)
        memory = tf.Variable(tf.constant(1e-6, shape=(batch_size, memory_length, memory_size)))
        memory = tf.stop_gradient(memory)

        return NTMState(ctrl_state=self.controller.h, read_list=read_list, w_list=w_list, memory=memory)

    @property
    def state_size(self):
        memory_length, memory_size = self.sizes[1:3]
        return NTMState(
            ctrl_state=1,#self.controller.state_size,
            read_list=[memory_length],
            w_list=[memory_size, memory_size],
            memory=tf.TensorShape([memory_size * memory_length]))

    @property
    def output_size(self):
        return self.sizes[4]
