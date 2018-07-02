# -*-coding:utf-8 -*-

import tensorflow as tf

with tf.variable_scope('scope1') as sp:
    var1 = tf.get_variable('v', [1])

with tf.variable_scope('scope2'):
    var2 = tf.get_variable('v', [1])

    with tf.variable_scope(sp) as sp1:
        var3 = tf.get_variable('v3', [1])


        with tf.variable_scope(''):
            var4 = tf.get_variable('v4', [1])


with tf.variable_scope('scope'):
    with tf.name_scope('bar'):
        v = tf.get_variable('v', [1])
        x = 1.0 + v
        with tf.name_scope(''):
            y = 1.0 + v
print('var4:', var4.name)
print('y.op:', y.op.name)
