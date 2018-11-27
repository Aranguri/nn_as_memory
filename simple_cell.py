import tensorflow as tf

class SimpleCell(tf.contrib.rnn.RNNCell):
    def __init__(self, input_size, output_size):
        self.w = tf.Variable(tf.random_normal((input_size, output_size)))
        self.output_size_ = output_size

    def __call__(self, x, prev_state):
        pop = tf.Print([0], [self.w[1][1]])
        with tf.control_dependencies([pop]):
            out = tf.matmul(x, self.w)
        return out, prev_state

    @property
    def state_size(self):
        return [0]

    @property
    def output_size(self):
        return self.output_size_
