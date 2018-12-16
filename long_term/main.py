'''
# Idea
Give a neural net several <word, definition> pairs. Then, we give it a query <?, definition'>  with definition' being another definition to refer to a word.
If it doesn't work, we can use definition = definition' (and limit the memory to be small)
To train, we can first start with a few steps between input and query. We then can enlarge those distances, and experiment with how good BPTT works.
Make somethign easy that works and build from there.
Output: vector in 50 dims. (Other possibility: char-by-char generator)

Baseline task: don't use <word, definition> at all. Just train an LSTM with word embeddings on the query <?, definition'> and measure the distance between the
 result of the query and the ground truth (to understand this distance, we can measure the distances between two random vectors, and the distances between the
 words in the definition and the ground truth word.) Also, we can measure the accuracy.

* we can try with making it possible to backprop through the word embeddings.
* what about BiRNNs?
* say we have one dataset of 100M images. Is it always the case that we want to train on new data? Or seeing the same data again could help?
* generate a large dataset
* try semantic average
* add this to a file
sudo -i
# sync; echo 1 > /proc/sys/vm/drop_caches
# sync; echo 2 > /proc/sys/vm/drop_caches
# sync; echo 3 > /proc/sys/vm/drop_caches
* make a single file for utils
'''

import tensorflow as tf
from task import DictTask
import itertools
from utils import *

h_size = 3
batch_size = 8
embeddings_size = 50

dict_task = DictTask(batch_size)
x = tf.placeholder(tf.float32, (batch_size, None, embeddings_size))
y = tf.placeholder(tf.float32, (batch_size, embeddings_size))

lstm = tf.contrib.rnn.LSTMCell(h_size)
lstm_output, _ = tf.nn.dynamic_rnn(lstm, x, dtype=tf.float32)
# lstm_output, _ = tf.nn.bidirectional_dynamic_rnn(lstm, x, dtype=tf.float32)
output = tf.layers.dense(lstm_output[:, -1], embeddings_size)

optimizer = tf.train.AdamOptimizer()
loss = tf.losses.mean_squared_error(output, y)
minimize = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in itertools.count():
        if i % 100 == 0:
            x_, y_ = dict_task.next_batch()
        output_, loss_, _ = sess.run([output, loss, minimize], feed_dict={x: x_, y: y_})
        # print(loss_)
        print(cosine_distance(output_, y_))
