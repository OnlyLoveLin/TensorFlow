# -*- coding:utf-8 -*-
import tensorflow as tf

tf.reset_default_graph()

w1 = tf.get_variable('w1', shape=[2])
w2 = tf.get_variable('w2', shape=[2])

w3 = tf.get_variable('w3', shape=[2])
w4 = tf.get_variable('w4', shape=[2])

y1 = w1 + w2 + w3
y2 = w3 + w4

# grad_ys 求梯度的输入值
gradients = tf.gradients([y1,y2], [w1,w2,w3,w4], grad_ys=[tf.convert_to_tensor([1.,2.]), tf.convert_to_tensor([3.,4.])])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print('w1:', sess.run(w1))
    print('w2:', sess.run(w2))
    print('w3:', sess.run(w3))
    print('w4:', sess.run(w4))
    print('gradients:', sess.run(gradients))
