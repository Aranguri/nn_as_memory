import tensorflow as tf
import collections
from utils import *

NTMState = collections.namedtuple('NTMState', ('ctrl_state', 'read', 'weights', 'memory', 'losses'))

class NTMCell(tf.contrib.rnn.RNNCell):
    def __init__(self, output_size, batch_size, memory_size, memory_length, memory_cell, h_size, shift_length):
        interface_size = memory_size + 1 + 1 + shift_length + 1
        params_size = 2 * interface_size + 2 * memory_size
        self.controller = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(h_size)])#update basic LSTM to newer type
        self.memory_cell = memory_cell
        self.sizes = [batch_size, memory_length, memory_size, shift_length, output_size, interface_size, params_size]

    def __call__(self, x, prev_state):
        #prepare input
        batch_size, memory_length, memory_size, _, output_size, interface_size, params_size = self.sizes
        ctrl_state_prev, read_prev, weights_prev, memory, losses = prev_state
        self.memory_cell.memory = memory
        x_and_r = tf.squeeze(tf.concat(([x], read_prev), axis=2))

        #execute controller
        ctrl_output, ctrl_state = self.controller(x_and_r, ctrl_state_prev)
        interface = tf.layers.dense(ctrl_output, params_size)
        interface = tf.clip_by_value(interface, -20, 20)
        interface = tf.split(interface, [interface_size, interface_size, memory_size, memory_size], axis=1)
        interface_read, interface_write, write, erase = interface

        #read head
        memory_prev = self.memory_cell.read()
        # pop = tf.Print([0], [memory_prev[0][0][0]])
        w_read = self.addressing(interface_read, memory_prev, weights_prev[0])
        read = expand(tf.einsum('ij,ijk->ik', w_read, memory_prev))

        #write head
        write, erase = tf.tanh(write), tf.sigmoid(erase)
        w_write = self.addressing(interface_write, memory_prev, weights_prev[1])
        self.memory_cell.write(w_write, write, erase)
        # loss = self.memory_cell.write(w_write, write, erase)
        # minimize = self.optimizer.minimize(loss)

        # prepare output
        c2o_input = tf.concat((ctrl_output, read_prev[0]), axis=1)
        output = tf.layers.dense(c2o_input, output_size)
        output = tf.clip_by_value(output, -20, 20)
        pop = tf.Print([0], [ca(w_read), ca(w_write), ca(interface_read), ca(interface_write), ca(weights_prev), ca(memory_prev)])
        #with tf.control_dependencies([pop]):
        weights = tf.concat((w_read, w_write), axis=0)

        return output, NTMState(ctrl_state=ctrl_state, read=read, weights=weights, memory=self.memory_cell.memory, losses=losses)

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
        w_tilde_num = tf.pow(tf.nn.relu(w_tilde), sharpener) #TODO: remove relu here. If we remove it, we get nan errors (eg tf.pow(-.4, 1.7) = nan). Somehow, we need to prevent w_tilde from having negative values.

        w = w_tilde_num / tf.reduce_sum(w_tilde_num, axis=1, keepdims=True)

        return w

    def zero_state(self, batch_size, dtype):
        batch_size, memory_length, memory_size = self.sizes[:3]
        ctrl_state = self.controller.zero_state(batch_size, dtype)
        read = tf.get_variable('read', initializer=tf.zeros_initializer(), shape=(1, batch_size, memory_size))
        weights = tf.get_variable('weights', initializer=tf.random_normal_initializer(stddev=1e-5), shape=(2 * batch_size, memory_length))
        memory = tf.get_variable('memory', initializer=tf.constant_initializer(1e-6), shape=(batch_size, memory_length, memory_size))
        losses = tf.get_variable('losses', initializer=tf.zeros_initializer(), shape=(1,))
        #memory = tf.stop_gradient(memory)

        return NTMState(ctrl_state=ctrl_state, read=read, weights=weights, memory=memory, losses=losses)

    @property
    def state_size(self):
        memory_length, memory_size = self.sizes[1:3]
        return NTMState(
            ctrl_state=self.controller.state_size,
            read=[memory_size],
            weights=[memory_length, memory_length],
            memory=tf.TensorShape([memory_length * memory_size]),
            losses=tf.TensorShape([1]))

    @property
    def output_size(self):
        return self.sizes[4]
