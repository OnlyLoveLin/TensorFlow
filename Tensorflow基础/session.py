# -*- coding:utf-8 _*_

import tensorflow as tf

a = tf.placeholder('float32')
b = tf.placeholder('float32')

add = tf.add(a, b)
mul = tf.multiply(a, b)

with tf.Session() as sess:
    print('相加:%i' % sess.run(add, feed_dict={a:3, b:4}))
    print('相乘:%i' % sess.run(mul, feed_dict={a:3, b:4}))
    print(sess.run([mul, add], feed_dict={a:3, b:4}))
