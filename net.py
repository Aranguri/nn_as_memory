import tensorflow as tf
import itertools
from utils import *
from ntm import NTMCell

batch_size = 32
input_size = output_size = 8

xs = tf.placeholder(tf.float32, shape=(batch_size, None, input_size + 1))
ys = tf.placeholder(tf.float32, shape=(batch_size, None, output_size))
output_length = tf.shape(ys)[1]

cell = NTMCell(output_size, batch_size, h_size=100, memory_length=128, memory_size=20, shift_length=3)
outputs, _ = tf.nn.dynamic_rnn(cell, xs, dtype=tf.float32)
outputs = outputs[:, -output_length:]

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=.9, beta2=.999)
loss = tf.losses.mean_squared_error(outputs, ys)
minimize = optimizer.minimize(loss)
'''
trainable_variables = tf.trainable_variables()
grads = tf.gradients(loss, trainable_variables)
print_op = tf.Print([0], [trainable_variables])
with tf.control_dependencies([print_op]):
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, trainable_variables), 50)
minimize = optimizer.apply_gradients(zip(grads, trainable_variables))
# grads = optimizer.compute_gradients(loss)
# clipped_grads = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in grads]
# minimize = optimizer.apply_gradients(clipped_grads)

binary_outputs = tf.to_float(outputs > .5)
errors = tf.equal(binary_outputs, ys)
accuracy = tf.reduce_mean(tf.to_float(errors))
cost = (1 - accuracy) * tf.to_float(output_length) * output_size
'''

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # tr_loss, tr_cost = {}, {}

    for i in itertools.count():
        bytes_length = 20#np.random.randint(1, 21)
        bytes = np.random.randint(0, 2, size=(batch_size, bytes_length, input_size))
        bytes_input = np.pad(bytes, ((0, 0), (0, 0), (0, 1)), 'constant')
        start_mark = np.zeros((batch_size, 1, input_size + 1))
        start_mark[:, :, -1] = 1
        x_ = np.concatenate((bytes_input, np.zeros_like(bytes_input)), axis=1)
        y_ = bytes

        outputs_, _ = sess.run([outputs, minimize], feed_dict={xs: x_, ys: y_})
        # print(outputs_[0][0], y_[0][0])
        #print(grads)
        #if i % 3 == 0:
        #    plot(tr_cost)

'''
#Next steps: assure the implementation is correct (up to inefficiencies,) by reading my code and benchmarking. Finish details in ntm.py
#why tanh for the key
#I'm assuming sequence number equals the number of individual sequences that the sentence saw. ie iteration_nums * batch_size
#logs
output_length = 20, num_seq = 1250 * 32, lr=1e-2, loss: .3 (other configs = ntm implementation)
#how to store all params?
1024: 64
5400: 38.5
'''
