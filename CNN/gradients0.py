# -*- coding:utf-8 -*-

import tensorflow as tf

w1 = tf.Variable([[1,2]])
w2 = tf.Variable([[3,4]])

y = tf.matmul(w1, [[9],[10]])
# 对w1求偏导
grads1 = tf.gradients(y, [w1])

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print('grads1:', sess.run(grads1))
