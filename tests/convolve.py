import tensorflow as tf

memory_length = 5
shift_range = 1
s = tf.constant([[0.9, 0.09, 0.01], [0.99, 0.009, 0.001]])
w_g = tf.constant([[1., 2, 3, 4, 5], [10., 20, 30, 40, 50]])

shift = tf.manip.roll(s, shift=shift_range, axis=1)
shift = tf.pad(shift, tf.constant([[0, 0,], [0, memory_length - (2*shift_range + 1)]]))
shift = tf.manip.roll(shift, shift=((-shift_range) % memory_length), axis=1)
t = tf.concat([tf.reverse(shift, axis=[1]), tf.reverse(s_, axis=[1])], axis=1)
rolled_matrix = tf.stack([t[:, memory_length - i - 1:memory_length * 2 - i - 1] for i in range(memory_length)], axis=1)
w_1 = tf.einsum('jik,jk->ji', rolled_matrix, w_g)

'''
s2 = tf.concat([s[:, :shift_range + 1],
               tf.zeros([s.get_shape()[0], memory_length - (shift_range * 2 + 1)]),
               s[:, -shift_range:]], axis=1)
t2 = tf.concat([tf.reverse(s2, axis=[1]), tf.reverse(s2, axis=[1])], axis=1)
s_matrix = tf.stack(
    [t2[:, memory_length - i - 1:memory_length * 2 - i - 1] for i in range(memory_length)],
    axis=1
)
w_ = tf.reduce_sum(tf.expand_dims(w_g, axis=1) * s_matrix, axis=2)      # eq (8)
'''

sess = tf.InteractiveSession()
print(sess.run(w_))
print(sess.run(w_1))
