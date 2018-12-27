import tensorflow as tf
from task import LoadedDictTask
import itertools
from utils import *

h_size = 200
batch_size = 32
embeddings_size = 50
max_length = 100
h1_size = 100
h2_size = 100

dict_task = LoadedDictTask(batch_size, num_batches=150)
x = tf.placeholder(tf.float32, (batch_size, max_length * embeddings_size))
y = tf.placeholder(tf.float32, (batch_size, embeddings_size))

'''
forward_lstm = tf.contrib.rnn.LSTMCell(h_size)
backward_lstm = tf.contrib.rnn.LSTMCell(h_size)
# lstm_output, _ = tf.nn.dynamic_rnn(forward_lstm, x, dtype=tf.float32)
_, lstm_output = tf.nn.bidirectional_dynamic_rnn(forward_lstm, backward_lstm, x, dtype=tf.float32)
lstm_output = tf.concat((lstm_output[0][0], lstm_output[1][0]), 1)
output = tf.layers.dense(lstm_output, embeddings_size)
'''

h1 = tf.layers.dense(x, h1_size, activation=tf.nn.relu)
h2 = tf.layers.dense(h1, h2_size, activation=tf.nn.relu)
output = tf.layers.dense(h2, embeddings_size)

optimizer = tf.train.AdamOptimizer(1e-4)
loss = tf.losses.mean_squared_error(output, y)
minimize = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    dev_dist = {}
    for i in itertools.count():
        x_, y_, label_ = dict_task.next_batch()
        padding = ((0, 0), (0, max_length - x_.shape[1]), (0, 0))
        x_ = np.pad(x_, padding, 'constant')
        x_ = x_.reshape(batch_size, -1)

        output_, loss_, _ = sess.run([output, loss, minimize], feed_dict={x: x_, y: y_})

        dist = np.mean([cosine_distance(v1, v2) for v1, v2 in zip(output_, y_)])
        print(f'Mean dist: {dist}')

        if i % 10 == 0:
            x_, y_, label_ = dict_task.dev_batch()
            padding = ((0, 0), (0, max_length - x_.shape[1]), (0, 0))
            x_ = np.pad(x_, padding, 'constant')
            x_ = x_.reshape(batch_size, -1)
            output_, loss_ = sess.run([output, loss], feed_dict={x: x_, y: y_})

            dev_dist[i//10] = np.mean([cosine_distance(v1, v2) for v1, v2 in zip(output_, y_)])
            print(f'DEV: mean dist: {dev_dist[i//10]}')
            plot(dev_dist)
