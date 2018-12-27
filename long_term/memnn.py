from task import LoadedDictTask
from utils import *

embeddings_size = 50
hidden_size = 50
task_batch_size = 32
mem_size = 8
batch_size = task_batch_size // mem_size
task = LoadedDictTask(task_batch_size, 140)

ws = tf.placeholder(tf.float32, (batch_size, mem_size, embeddings_size))
ds = tf.placeholder(tf.float32, (batch_size, mem_size, None, embeddings_size))
wq = tf.placeholder(tf.int32, (batch_size, mem_size))
dq = tf.placeholder(tf.float32, (batch_size, None, embeddings_size))

trans_ds = tf.layers.dense(ds, hidden_size, use_bias=False) #check: do we transform every entry in the matrix?
trans_dq = tf.layers.dense(dq, hidden_size, use_bias=False)
similarity = tf.einsum('ijkl,ikl->ij', trans_ds, trans_dq)

correct_cases = tf.equal(tf.argmax(similarity, 1), tf.argmax(wq, 1))
accuracy = tf.reduce_mean(tf.to_float(correct_cases))

loss = tf.losses.softmax_cross_entropy(wq, similarity)
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(loss)

def next_batch_wrapper(train=True):
    defs1, defs2, words, (m1, m2, w) = task.next_batch() if train else task.dev_batch()

    defs1 = defs1.reshape(batch_size, mem_size, -1, embeddings_size)
    defs2 = defs2.reshape(batch_size, mem_size, -1, embeddings_size)
    words = words.reshape(batch_size, mem_size, embeddings_size)
    m1 = np.array(m1).reshape(batch_size, mem_size)
    m2 = np.array(m2).reshape(batch_size, mem_size)
    query_ixs = np.random.randint(mem_size, size=(batch_size,))
    query_words = one_hot(query_ixs, mem_size)
    query_defs = defs2[range(batch_size), query_ixs]

    return {ws: words, ds: defs1, wq: query_words, dq: query_defs}, m1, m2

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tr_acc, dev_loss, dev_acc = {}, {}, {}

    for i in itertools.count():
        feed_dict, _, _ = next_batch_wrapper()
        tr_acc[i], similarity_, _ = sess.run([accuracy, similarity, minimize], feed_dict)

        if i % 25 == 0:
            feed_dict, defs1, defs2 = next_batch_wrapper(train=False)
            dev_loss[i//25], similarity_, dev_acc[i//25] = sess.run([loss, similarity, accuracy], feed_dict)

            # similarity_ = np.expand_dims(np.argmax(similarity_, 1), 1)
            correct = np.argmax(similarity_, 1) == feed_dict[wq]
            ps(correct)

            for batch_d1, batch_d2, batch_c in zip(defs1, defs2, correct):
                for d1, d2, c in zip(batch_d1, batch_d2, batch_c):
                    print(''.join(['_']*100))
                    print(d1)
                    print(d2)
                    print(c)

            print('tr acc', dict_mean(tr_acc), tr_acc[i//25])
            print('devacc', dict_mean(dev_acc), dev_acc[i//25])
            # smooth_plot(tr_acc, 300)
            smooth_plot(dev_acc, 300)


'''

# probs = tf.nn.softmax(similarity)
# wq_guess = tf.einsum('ij,ijl->il', probs, ws)
#do we want to transform wq_guess?'''
