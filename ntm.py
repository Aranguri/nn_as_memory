import tensorflow as tf
import collections
from utils import *

NTMState = collections.namedtuple('NTMState', ('ctrl_state', 'read', 'weights', 'memory'))

class NTMCell(tf.contrib.rnn.RNNCell):
    def __init__(self, output_size, batch_size, h_size, memory_size, memory_length, shift_length):
        interface_size = memory_size + 1 + 1 + shift_length + 1
        params_size = 2 * interface_size + 2 * memory_size
        self.controller = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(h_size)])
        self.sizes = [batch_size, memory_length, memory_size, shift_length, output_size, interface_size, params_size]

    def __call__(self, x, prev_state):
        #prepare input
        batch_size, memory_length, memory_size, _, output_size, interface_size, params_size = self.sizes
        ctrl_state_prev, read_prev, weights_prev, memory_prev = prev_state
        x_and_r = tf.squeeze(tf.concat(([x], read_prev), axis=2))

        #execute controller
        ctrl_output, ctrl_state = self.controller(x_and_r, ctrl_state_prev)
        interface = tf.layers.dense(ctrl_output, params_size)
        interface = tf.clip_by_value(interface, -20, 20)
        interface = tf.split(interface, [interface_size, interface_size, memory_size, memory_size], axis=1)
        interface_read, interface_write, write, erase = interface

        #read head
        w_read = self.addressing(interface_read, memory_prev, weights_prev[0])
        read = expand(tf.einsum('ij,ijk->ik', w_read, memory_prev))

        #write head
        write, erase = tf.tanh(write), tf.sigmoid(erase)
        w_write = self.addressing(interface_write, memory_prev, weights_prev[1])
        memory = memory_prev * (1 - outer_prod(w_write, erase)) + outer_prod(w_write, write)

        # prepare output
        c2o_input = tf.concat((ctrl_output, read_prev[0]), axis=1)
        output = tf.layers.dense(c2o_input, output_size)
        output = tf.clip_by_value(output, -20, 20)
        weights = tf.concat((w_read, w_write), axis=0)

        return output, NTMState(ctrl_state=ctrl_state, read=read, weights=weights, memory=memory)

    def addressing(self, interface, m_prev, w_prev):
        # prepare input
        batch_size, memory_length, memory_size, shift_length = self.sizes[:4]
        key, gate, b, shift, sharpener = tf.split(interface, [memory_size, 1, 1, shift_length, 1], axis=1)
        key, gate, b = tf.tanh(key), tf.sigmoid(gate), tf.nn.softplus(b)
        shift, sharpener = tf.nn.softmax(shift), (tf.nn.softplus(sharpener) + 1)
        shift = tf.pad(shift, tf.constant([[0, 0,], [0, memory_length - shift_length]]))

        # gate between content-based weight and previous weight
        unnorm_similarity = tf.einsum('ik,ijk->ij', key, m_prev)
        similarity = unnorm_similarity / (tf.norm(m_prev, axis=2) * tf.norm(key, axis=1, keepdims=True) + 1e-8)
        w_c = tf.nn.softmax(b * similarity)
        w_g = gate * w_c + (1 - gate) * w_prev

        # convolve
        shift_range = (shift_length - 1) // 2
        pad = tf.zeros((batch_size, memory_length - shift_length))
        shift = tf.concat([shift[:, :shift_range + 1], pad, shift[:, -shift_range:]], axis=1)
        shift_matrix = tf.concat([tf.reverse(shift, axis=[1]), tf.reverse(shift, axis=[1])], axis=1)
        rolled_matrix = tf.stack([shift_matrix[:, memory_length - i - 1:memory_length * 2 - i - 1]
                                  for i in range(memory_length)], axis=1)
        w_tilde = tf.einsum('jik,jk->ji', rolled_matrix, w_g)
        w_tilde_num = tf.pow(w_tilde, sharpener)
        w = w_tilde_num / tf.reduce_sum(w_tilde_num, axis=1, keepdims=True)

        return w

    def zero_state(self, batch_size, dtype):
        batch_size, memory_length, memory_size = self.sizes[:3]
        ctrl_state = self.controller.zero_state(batch_size, dtype)
        read = tf.Variable(tf.constant(0.0, shape=(1, batch_size, memory_size)))
        weights = tf.Variable(tf.random_normal(shape=(2 * batch_size, memory_length), stddev=1e-5))
        memory = tf.Variable(tf.constant(1e-6, shape=(batch_size, memory_length, memory_size)))
        #memory = tf.stop_gradient(memory)

        return NTMState(ctrl_state=ctrl_state, read=read, weights=weights, memory=memory)

    @property
    def state_size(self):
        memory_length, memory_size = self.sizes[1:3]
        return NTMState(
            ctrl_state=self.controller.state_size,
            read=[memory_size],
            weights=[memory_length, memory_length],
            memory=tf.TensorShape([memory_size * memory_length]))

    @property
    def output_size(self):
        return self.sizes[4]
