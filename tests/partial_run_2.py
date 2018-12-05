import tensorflow as tf
import collections

m = tf.Variable(tf.constant(1.))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

class Partial(tf.contrib.rnn.RNNCell):
    def __init__(self):
        pass

    def __call__(self, x, state):
        h, w = state
        w = h * x
        h = h * x
        m_prime = m + h
        loss = m_prime - m
        minimize = optimizer.minimize(loss)
        return h, (h, w)

    def zero_state(self, batch_size, dtype):
        h = tf.Variable(tf.constant(1e-6, shape=(1, 1)))
        w = tf.Variable(tf.constant(1., shape=(1, 1)))
        return (h, w)

    @property
    def state_size(self):
        return 1.

    @property
    def output_size(self):
        return 1.

xs = tf.placeholder(tf.float32, shape=(1, 3, 1))
cell = Partial()
outputs, final_state = tf.nn.dynamic_rnn(cell, xs, dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    xs_ = [[[1.], [2.], [3.]]]
    print(sess.run(outputs, feed_dict={xs: xs_}))
