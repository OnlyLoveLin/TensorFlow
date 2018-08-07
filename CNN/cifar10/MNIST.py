# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 200

mnist = input_data.read_data_sets('./data/', one_hot=True)

# 定义权重的函数
def weight_variable(shape):
    inital = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(inital)


# 定义偏置的函数
def bias_variable(shape):
    inital = tf.constant(0.1, shape=shape)
    return tf.Variable(inital)


# 定义卷积的函数
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')


# 定义一个最大池化函数
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


# 定义一个平均池化函数
def avg_pool_7x7(x):
    return tf.nn.avg_pool(x, ksize=[1,7,7,1], strides=[1,7,7,1], padding='SAME')


# 定义占位符
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

W_conv1 = weight_variable([5,5,1,64])
b_conv1 = bias_variable([64])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5,5,64,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([5,5,64,10])
b_conv3 = bias_variable([10])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

nt_pool3 = avg_pool_7x7(h_conv3)

nt_pool3_flat = tf.reshape(nt_pool3, [-1,10])

y_conv = tf.nn.softmax(nt_pool3_flat)

cross_entropy = -tf.reduce_sum(y*tf.log(y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,-1), tf.argmax(y,-1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))


sess = tf.Session()
sess.run([tf.global_variables_initializer()])
tf.train.start_queue_runners(sess=sess)
for i in range(10000):
    xs, ys = mnist.train.next_batch(batch_size)
    xs = np.reshape(xs, (batch_size,28,28,1))
    train_step.run(feed_dict={x:xs, y:ys}, session=sess)
    if i % 200:
        train_accuracy = accuracy.eval(feed_dict={x:xs, y:ys}, session=sess)
        print('After %d training step, training accuracy is %g' %(i, train_accuracy))

