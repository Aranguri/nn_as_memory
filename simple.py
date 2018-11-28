import numpy as np
from simple_cell import SimpleCell
from ntm import NTMCell
import tensorflow as tf

batch_size = 4
input_size = output_size = 2

xs = tf.placeholder(tf.float32, shape=(4, 3, 2))
ys = tf.placeholder(tf.float32, shape=(4, 3, 2))
cell = NTMCell(output_size, batch_size, h_size=100, memory_length=128, memory_size=20, shift_length=3)
output, _ = tf.nn.dynamic_rnn(cell, xs, dtype=tf.float32)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
loss = tf.losses.mean_squared_error(output, ys)
minimize = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10000):
        xs_ = np.arange(24).reshape(4, 3, 2)
        ys_ = np.arange(24).reshape(4, 3, 2)
        output_, _ = sess.run([output, minimize], feed_dict={xs: xs_, ys: ys_})
        # print(output_[0, 0])
