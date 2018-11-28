import tensorflow as tf
import itertools
from utils import *
from ntm import NTMCell

batch_size = 32
input_size = output_size = 8
grad_max_norm = 50

xs = tf.placeholder(tf.float32, shape=(batch_size, None, input_size + 1))
ys = tf.placeholder(tf.float32, shape=(batch_size, None, output_size))

cell = NTMCell(output_size, batch_size, h_size=100, memory_size=20, memory_length=128, shift_length=3)
outputs, _ = tf.nn.dynamic_rnn(cell, xs, dtype=tf.float32)
outputs = outputs[:, -tf.shape(ys)[1]:]

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=ys, logits=outputs)
trainable_vars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, trainable_vars), grad_max_norm)
minimize = optimizer.apply_gradients(zip(grads, trainable_vars))

binary_outputs = tf.to_float(outputs > 0)
errors = tf.not_equal(binary_outputs, ys)
cost = tf.reduce_sum(tf.to_float(errors)) / batch_size

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tr_loss, tr_cost = {}, {}

    for i in itertools.count():
        # generate task
        bytes_length = 20
        bytes = np.random.randint(0, 2, size=(batch_size, bytes_length, input_size))
        bytes_input = np.pad(bytes, ((0, 0), (0, 0), (0, 1)), 'constant')
        start_mark = np.zeros((batch_size, 1, input_size + 1))
        start_mark[:, :, -1] = 1
        x_ = np.concatenate((bytes_input, np.zeros_like(bytes_input)), axis=1)
        y_ = bytes

        # feed task to the ntm
        tr_loss[i], tr_cost[i], outputs_, _ = sess.run([loss, cost, binary_outputs, minimize], feed_dict={xs: x_, ys: y_})
        last = list(tr_cost.values())[-10:]
        print(i, np.mean(last))
        #plot(tr_cost)

'''
Thigns to try
* is the clip_by_global_nrom doing something? print it
* test my lstm
* use a better cost (avg over last 5)

#Next steps: assure the implementation is correct (up to inefficiencies,) by reading my code and benchmarking. Finish details in ntm.py
#why tanh for the key
#I'm assuming sequence number equals the number of individual sequences that the sentence saw. ie iteration_nums * batch_size
#logs
output_length = 20, num_seq = 1250 * 32, lr=1e-2, loss: .3 (other configs = ntm implementation)
#how to store all params?
1024: 64
5400: 38.5

memory_size -> memory_length (128)
num_vector_dim -> memory_size (20)

Their task, their code @150 .72 @200 .72. @150 .70 @200 .67
My task, their code: @200 .76. New @150 .75 @200 .75. New @150 .76  @200 .75.5
My interface, their task and code: @150 .75 @200 .76
My interface, their task and code: @130 .71 (changed loss) @130 .73 @150 .72 (used max norm)
My interface, my task, their code: @130 74.5 @150 .72 @200 69.5 @230 67.5
What do we know shofar? My minimization and task are ok. My initialization of code is xxxx. Now going to check my ntm as a whole.
My interface and task, their code (but dirty:): @130 76 and 72 @150 73 and 73 @200 .72 and 71 @230 .71 and 72
Everything mine: @130 .74 @150 73.5 @200 73 @230 71.5
Everything mine (lr 1e-3): 57
Everything their: @3000 53
Everything mine (updated code): @130 72 @150 74 @230 70.5
Changing to dense:              @130 74 @150 73 @230 70
Both: @520 around 63
@150 72 @200 70
Performance seems to be decreasing a

'''
