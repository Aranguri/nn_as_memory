'''
Next tasks:
The goal is to have an NTM that we know it works to start adding things to it. Otherwise,
  it's difficult to debug the things we are adding.
* Conclude something from the two experiments. How can I verify that the official NTM is better than a LSTM. After that, how can I verify that my NTM is similar to official NTM and better than LSTM.
* Errands: Order all the tasks and things to do; get credits for GPU.
It doesn't make sense to continue adding features if I don't get this right. For the good thing about coding new features is that they become real, but they aren't real if the codebase isn't working.

Experiment:
the difference isn't clear at all (there may be no difference) in the first steps (in the first 2000 iterations). It's only after the 2000 iterations that the NTM starts to outperform the NTM by a lot.
(around 1000 they are the same|)
LSTM ... @5200: 44
LSTM in copy (stand) @31000: 33.5
NTM in copy (stand) @5200: 0.03
'''


from pprint import pprint
import tensorflow as tf
import itertools
from utils import *
from NeuralTuringMachine.ntm import NTMCell
from memory_cell_nn import MemoryNN
from memory_cell_matrix import MemoryMatrix
from poly_task import PolyTask

batch_size = 128
input_size = 2
output_size = 1
input_length = 64
output_length = input_length
grad_max_norm = 50
memory_size = 20
basis_length = 20
h_lengths = [basis_length, 8, 16, 32, 64]
memory_length = 128
mann = 'official'
h_lengths_LSTM = [100]#, 50]

basis = tf.get_variable('basis', (batch_size, basis_length, memory_size))
memory_cell = MemoryMatrix(basis, h_lengths, memory_length, memory_size, batch_size)

if mann == 'official':
    cell = NTMCell(1, 8, memory_length, memory_size,
                    1, 1, addressing_mode='content_and_location',
                    shift_range=1, reuse=False, output_dim=1,
                    clip_value=20)
elif mann == 'mine':
    cell = NTMCell(output_size, batch_size, memory_size, memory_length, memory_cell, h_size=100, shift_length=3)

else:
    layers = [tf.contrib.rnn.LSTMBlockCell(num_units=h) for h in h_lengths_LSTM]
    cell = tf.nn.rnn_cell.MultiRNNCell(layers)

xs = tf.placeholder(tf.float32, shape=(batch_size, input_length, input_size))
ys = tf.placeholder(tf.float32, shape=(batch_size, input_length, output_size))
outputs, _ = tf.nn.dynamic_rnn(cell, xs, dtype=tf.float32)
outputs = tf.layers.dense(outputs, output_size)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
loss = tf.losses.mean_squared_error(ys, outputs)
trainable_vars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients([loss], trainable_vars), grad_max_norm) #{why|is this useful}
minimize = optimizer.apply_gradients(zip(grads, trainable_vars))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tr_loss, tr_cost = {}, {}
    poly_task = PolyTask(batch_size, input_length)

    for i in itertools.count():
        # feed task to the ntm
        x_, y_ = poly_task.next_batch()
        tr_loss[i], outputs_, _ = sess.run([loss, outputs, minimize], feed_dict={xs: x_, ys: y_})
        outputs_ = outputs_.reshape(batch_size, input_length)
        y_ = y_.reshape(batch_size, input_length)

        last = list(tr_loss.values())[-10:]
        # pprint(list(zip(x_[0,:,0], x_[0,:,1], y_[0], outputs_[0], abs(outputs_[0] - y_[0]))), width=200)
        if i % 100 == 0:
            np.save(f'{mann}1', tr_loss)
        print(i, np.mean(last))
        plot(tr_loss)



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

Instead of training the ax**2+bx+c from scratch, try curriculuumum learning

Notice that we changed the loss from sigmoid cross entropy {what was that?} to mse. The inverse change gives us a lot of perforamnce ~~beware.
why are there times in training an algorithm that all of a sudden the loss decreases by a lot (and it seems to happen around the same iteration often)

Look for a correlation between the numbers a, b, and c of the polynomial and what's stored in the memory.

    # generate task
    # bytes_length = output_length
    # bytes = np.random.randint(0, 2, size=(batch_size, bytes_length, repetitions * input_size))
    # bytes_input = np.pad(bytes, ((0, 0), (0, 0), (0, 1)), 'constant')
    # start_mark = np.zeros((batch_size, 1, input_size + 1))
    # start_mark[:, :, -1] = 1
    # x_ = np.concatenate((bytes, np.zeros_like(bytes)), axis=1)
    # y_ = bytes
    # x_ = np.array(x_, dtype=np.float32)

next steps
* compare to a vainilla LSTM


PolyTask
mann = True, input_length = 64, lr = 1e-2
100: 3. 300: .3
mann = False
Comparable performance
mann = False, batch_size = 128

it would be great to have a command to copy the last n commands (using the top arrow n * n times bothers me)
'''
