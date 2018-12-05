import tensorflow as tf

i = tf.constant(0)
w = tf.Variable(tf.constant(1e-6))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

def cond(j, v):
    return tf.less(j, 3)

def loop(j, v):
    minimize = optimizer.minimize(v)
    return tf.add(j, 1), v

run = tf.while_loop(cond, loop, [i, w], shape_invariants=[i.get_shape(), w.get_shape()])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(run))
