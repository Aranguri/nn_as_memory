import tensorflow as tf
import itertools
from utils import *
from lstm_layer import LSTM

learning_rate = 1e-1
beta1 = 0.9
beta2 = 0.999
batch_size = 32

memory_length = 128
memory_size = 20
bits_length = 7
input_length = output_length = 2 * bits_length
input_size = output_size = 4
h_size = 4

xs = tf.placeholder(tf.float32, shape=(batch_size, input_length, input_size))
ys = tf.placeholder(tf.float32, shape=(batch_size, output_length, output_size))
initial_memories = tf.Variable(tf.constant(1e-6, shape=(1, batch_size, memory_length, memory_size)))
initial_memories = tf.stop_gradient(initial_memories) #Avoid backprop on initial memory
initial_w_prevs = tf.random_normal(shape=(2 * batch_size, memory_length), stddev=1e-5)

interface_size = memory_size + 1 + 1
ctrl_size = 2 * interface_size + 2 * memory_size + output_size
controller = LSTM(tf.zeros([2, batch_size, h_size]), ctrl_size)

def io_head(interface, memory, w_prev):
    # prepare input
    key, gate, b = tf.split(interface, [memory_size, 1, 1], axis=1)
    key = tf.tanh(key)
    gate = tf.sigmoid(gate)
    b = tf.transpose(tf.nn.softplus(b))
    memory = tf.reshape(memory, (memory_length, batch_size, memory_size))
    #todo: add convolve

    def similarity(vec): #todo: implement a matrix version of this.
        return tf.reduce_sum(vec * key, axis=1) / (tf.norm(vec, axis=1) * tf.norm(key, axis=1) + 1e-8)

    #gate(content-based weight, previous weight)
    unnorm_w_c = tf.exp(b * tf.map_fn(similarity, memory))
    unnorm_w_c = tf.transpose(unnorm_w_c)
    w_c = unnorm_w_c / tf.reduce_sum(unnorm_w_c, axis=1, keepdims=True)
    w_g = gate * w_c + (1 - gate) * w_prev
    w_g = tf.expand_dims(w_g, 2)
    return w_g

def cond(outputs, memories, r, ctrl_h, w_prevs):
    return tf.reduce_any(tf.not_equal(tf.shape(outputs), tf.shape(ys)))

def body(outputs, memories, r, ctrl_h, w_prevs):
    #prepare input
    controller.h = ctrl_h
    w_read_prev, w_write_prev = tf.split(w_prevs, [batch_size, batch_size], axis=0)
    x = xs[:, tf.shape(outputs)[1]]
    x = tf.reshape(x, shape=(batch_size, input_size))#todo: is this necessary?
    memory = memories[-1]

    #execute controller
    x_and_r = tf.concat((x, r), axis=1)
    interface = tf.split(controller(x_and_r), [interface_size, interface_size, memory_size, memory_size, output_size], axis=1)
    interface_read, interface_write, write, erase, output = interface
    output = tf.expand_dims(output, 1)

    #read head
    w_read = io_head(interface_read, memory, w_read_prev)
    r = tf.reduce_sum(w_read * memory, axis=1)

    #write head
    write = tf.tanh(write)
    write = tf.reshape(write, (batch_size, 1, memory_size))
    erase = tf.sigmoid(erase)
    erase = tf.reshape(erase, (batch_size, 1, memory_size))
    w_write = io_head(interface_write, memory, w_write_prev)
    memory = memory * (1 - w_write * erase) + w_write * write
    memory = tf.reshape(memory, (1, batch_size, memory_length, memory_size))

    # algo
    memories = tf.concat((memories, memory), axis=0)
    outputs = tf.concat((outputs, output), axis=1) #todo: why axis=0?
    w_prevs = tf.concat((w_read, w_write), axis=0)
    w_prevs = tf.reshape(w_prevs, (2*batch_size, memory_length))

    print_op = tf.Print([memory], [tf.shape(w_write)])
    with tf.control_dependencies([print_op]):
        memories = tf.identity(memories)

    return outputs, memories, r, controller.h, w_prevs

#is there a better way of doing this?
shapes = [tf.TensorShape([batch_size, None, output_size]), tf.TensorShape([None, batch_size, memory_length, memory_size]),
          tf.TensorShape([batch_size, memory_size]), tf.TensorShape([2, batch_size, h_size]), tf.TensorShape([2 * batch_size, memory_length])]
last_state = tf.while_loop(cond, body, [tf.constant(0.0, shape=(batch_size, 0, output_size)), initial_memories, tf.constant(0.0, shape=(batch_size, memory_size)), controller.h, initial_w_prevs], shape_invariants=[*shapes])
output, memories = last_state[0:2]

optimizer = tf.train.AdamOptimizer(learning_rate, beta1, beta2)
loss = tf.losses.mean_squared_error(output, ys)
minimize = optimizer.minimize(loss)

tf.summary.scalar('loss', loss)
tr_loss = {}

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('./logs/9/train ', sess.graph)

    for i in itertools.count():
        bits = np.random.randn(batch_size, bits_length, input_size)
        x_ = np.concatenate((bits, np.zeros_like(bits)), axis=1)
        y_ = np.concatenate((np.zeros_like(bits), bits), axis=1)
        tr_loss[i], _, output_, memories_ = sess.run([loss, minimize, output, memories], feed_dict={xs: x_, ys: y_})
        #print(memories_[-1][0][0])
        # print('Loss', tr_loss[i])
        #if i > 100:
        #   print(f'{i} loss {np.mean([tr_loss[len(tr_loss) - j] for j in range(1, 100)])}')
        # print(output_[0], ys[0])

        if i % 25 == 0:
            merge = tf.summary.merge_all()
            memories_, summary, output_, initial_memories_ = sess.run([memories, merge, output, initial_memories], feed_dict={xs: x_, ys: y_})
            # train_writer.add_summary(summary, i)
            #print([m[0][0] for m in memories_])
            #ps(memories_)
            #print(memories_[-1][0][0])
            plot(tr_loss)
            '''
            # Debugging
            print('Prediction', out[0])
            print('Real', ys[0])
            print('')
            plt.ion()
            plt.cla()
            plt.imshow(memories_[:-1, 0, :])
            plt.pause(1e-8)
            '''
