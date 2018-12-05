import tensorflow as tf
import itertools
from utils import *
from ntm import NTMCell
from memory_cell_nn import MemoryNN
from memory_cell_matrix import MemoryMatrix
from poly_task import PolyTask

batch_size = 32
repetitions = 1
input_size = output_size = 1
input_length = 16
output_length = input_length# // 2
grad_max_norm = 50
memory_size = 20
basis_length = 4
h_lengths = [basis_length, 8, 16, 32, 64]
memory_length = 128

basis = tf.get_variable('basis', (batch_size, basis_length, memory_size))
memory_cell = MemoryMatrix(basis, h_lengths, memory_length, memory_size, batch_size)
cell = NTMCell(output_size, batch_size, memory_size, memory_length, memory_cell, h_size=100, shift_length=3)

xs = tf.placeholder(tf.float32, shape=(batch_size, input_length, repetitions * input_size))
ys = tf.placeholder(tf.float32, shape=(batch_size, input_length))

def cond(outputs):
    # tf.matmul(outputs, ys)
    return tf.not_equal(tf.shape(outputs)[1], tf.shape(ys)[1])

def body(outputs):
    step = tf.shape(outputs)[1]
    x = xs[:, :, step:step + output_size]
    x = tf.reshape(x, shape=[batch_size, input_length, input_size])
    output, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    output = output[:, -output_length:]
    outputs = tf.concat((outputs, output), axis=2)
    return outputs

loop_var = [tf.constant(0., shape=(batch_size, output_length))]
loop_shape = [tf.TensorShape((batch_size, output_length))]
outputs = tf.while_loop(cond, body, loop_var, shape_invariants=loop_shape)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=ys, logits=outputs)
trainable_vars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients([loss], trainable_vars), grad_max_norm)
minimize = optimizer.apply_gradients(zip(grads, trainable_vars))

binary_outputs = tf.to_float(outputs > 0)
errors = tf.not_equal(binary_outputs, ys)
cost = tf.reduce_sum(tf.to_float(errors)) / batch_size

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tr_loss, tr_cost = {}, {}
    poly_task = PolyTask(batch_size, input_length)

    for i in itertools.count():
        # generate task
        # bytes_length = output_length
        # bytes = np.random.randint(0, 2, size=(batch_size, bytes_length, repetitions * input_size))
        # bytes_input = np.pad(bytes, ((0, 0), (0, 0), (0, 1)), 'constant')
        # start_mark = np.zeros((batch_size, 1, input_size + 1))
        # start_mark[:, :, -1] = 1
        # x_ = np.concatenate((bytes, np.zeros_like(bytes)), axis=1)
        # y_ = bytes
        # x_ = np.array(x_, dtype=np.float32)

        # feed task to the ntm
        x_, y_ = poly_task.next_batch()
        ps(x_, y_)
        # b = sess.run([a], feed_dict={xs: x_, ys: y_})
        tr_loss[i], tr_cost[i], outputs_, _ = sess.run([loss, cost, binary_outputs, minimize], feed_dict={xs: x_, ys: y_})
        last = list(tr_cost.values())[-10:]
        print(i, np.mean(last))
        plot(tr_cost)



'''
Thigns to try
* is the clip_by_global_nrom doing something? print it
* test my lstm
* use a better cost (avg over last 5)
* assure that the tf.layers.dense are working as desired

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
Mine (nn-for-m): @150 72  @200 70
Mine (nn-for-m with 2 cost funs): @150 74 @200 71

Dude!
Now: polynomials. the idea is to test whether the nn can learn to predict the next value of the polynomial.  each 16 steps we generate a new polynomial. Test this using just the short-term memory.
Then we can add a long-term memory that can be useful for the following task. Repeated polynomials (every 100 steps or so, the polynomials start to repeat, but in a random way.)

#init short-term memory with a reading from the long-term memoryself.
# we can do this by giving the long-term memory as a param for the zero_state
# or to make it remain as an attrbute in ntm. ie we assign self.long-term_mem = somethign
# in the first call (line 20) and then in zero state we read from it. :)s

There are several things to do. The thing is that when I'm not working is difficult to see them.
* First, short-term memory could be a neural net. It could be either a neural net where you edit the last layer or a neural net that uses the key as input or something like unixpicle
* Second, long-term short-term interaction with matrices, or with nn.
* third, try the wikidata task.

'''
