# -*- coding:utf-8 -*-
import tensorflow as tf

tf.reset_default_graph()

w1 = tf.get_variable('w1', shape=[2])
w2 = tf.get_variable('w2', shape=[2])

w3 = tf.get_variable('w3', shape=[2])
w4 = tf.get_variable('w4', shape=[2])

y1 = w1 + w2 + w3
y2 = w3 + w4

# 设置梯度停止
a = w1 + w2
a_stoped = tf.stop_gradient(a)
y3 = a_stoped + w3

# grad_ys 求梯度的输入值
gradients1 = tf.gradients([y1,y2], [w1,w2,w3,w4], grad_ys=[tf.convert_to_tensor([1.,2.]), tf.convert_to_tensor([3.,4.])])

gradients2 = tf.gradients(y3, [w1,w2,w3], grad_ys=tf.convert_to_tensor([1.,2.]))

gradients3 = tf.gradients(y3, [w3], grad_ys=tf.convert_to_tensor([1.,2.]))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print('w1:', sess.run(w1))
    print('w2:', sess.run(w2))
    print('w3:', sess.run(w3))
    print('w4:', sess.run(w4))
    print('gradients1:', sess.run(gradients1))
    # print('gradients2:', sess.run(gradients2))
    # 程序试图去求一个None的梯度，所以报错,[None, None, <tf.Tensor 'gradients_1/add_4_grad/Reshape_1:0' shape=(2,) dtype=float32>]
    print('gradients3:', sess.run(gradients3))
