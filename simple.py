import numpy as np
from simple_cell import SimpleCell
import tensorflow as tf

output_size = 2
xs = tf.placeholder(tf.float32, shape=(4, 3, 2))
ys = tf.placeholder(tf.float32, shape=(4, 3, 2))
cell = SimpleCell(tf.shape(xs)[2], output_size)
output, _ = tf.nn.dynamic_rnn(cell, xs, dtype=tf.float32)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-1)
loss = tf.losses.mean_squared_error(output, ys)
minimize = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    xs_ = np.arange(24).reshape(4, 3, 2)
    ys_ = np.arange(24).reshape(4, 3, 2)
    for _ in range(10000):
        output_, _ = sess.run([output, minimize], feed_dict={xs: xs_, ys: ys_})
        # print(output_[0, 0])
