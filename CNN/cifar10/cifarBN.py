# -*- coding:utf-8 -*-

from tensorflow.contrib.layers.python.layers import batch_norm
import cifar10_input
import tensorflow as tf
import numpy as np
import time


batch_size = 128
data_dir = 'cifar-10-batches-bin/'
# 准备数据
print('begin')
images_train, labels_train = cifar10_input.inputs(eval_data=False, data_dir=data_dir, batch_size=batch_size)
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)
print('begin data')

# 定义权重的函数
def weight_variable(shape):
    inital =  tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(inital)


# 定义偏置的函数
def bias_variable(shape):
    inital = tf.constant(0.1, shape=shape)
    return tf.Variable(inital)


# 定义卷积的函数
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')


# 定义最大池化函数
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


# 定义均值池化的函数
def avg_pool_6x6(x):
    return tf.nn.avg_pool(x, ksize=[1,6,6,1], strides=[1,6,6,1], padding='SAME')


# BN函数
def batch_norm_layer(value, train=None, name='batch_normal'):
    if train is not None:
        return batch_norm(value, decay=0.9, updates_collections=None, is_training=True)
    else:
        return batch_norm(value, decay=0.9, updates_collections=None, is_training=True)


# 定义占位符
x = tf.placeholder(tf.float32, [None, 24 , 24, 3])
y = tf.placeholder(tf.float32, [None, 10])

# 定义一个训练状态
train = tf.placeholder(tf.float32)

W_conv1 = weight_variable([5, 5, 3, 64])
b_conv1 = bias_variable([64])

x_image = tf.reshape(x, [-1, 24, 24, 3])

# 定义第一层卷积层,添加BN层
h_conv1 = tf.nn.relu(batch_norm_layer((conv2d(x_image, W_conv1) + b_conv1), train))
# 定义第一层池化层
h_pool1 = max_pool_2x2(h_conv1)


W_conv21 = weight_variable([5, 1, 64, 64])
b_conv21 = bias_variable([64])

h_conv21 = tf.nn.relu(batch_norm_layer((conv2d(h_pool1, W_conv21) + b_conv21), train))
h_pool21 = max_pool_2x2(h_conv21)

W_conv22 = weight_variable([5, 5, 64, 64])
b_conv22 = bias_variable([64])

h_conv22 = tf.nn.relu(batch_norm_layer((conv2d(h_pool21, W_conv22) + b_conv22), train))
h_pool22 = max_pool_2x2(h_conv22)


W_conv3 = weight_variable([5, 5, 64, 10])
b_conv3 = bias_variable([10])

h_conv3 = tf.nn.relu(conv2d(h_pool22, W_conv3) + b_conv3)

# 进行全局平均池化
nt_pool3 = avg_pool_6x6(h_conv3)

nt_pool3_flat = tf.reshape(nt_pool3, [-1, 10])

# 进行softmax分类
y_conv = tf.nn.softmax(nt_pool3_flat)

# 交叉熵
cross_entropy = -tf.reduce_sum(y*tf.log(y_conv))

# 退化学习率
global_step = tf.Variable(0, trainable=False)
decaylearning_rate = tf.train.exponential_decay(0.04, global_step, 1000, 0.9)

train_step = tf.train.AdamOptimizer(decaylearning_rate).minimize(cross_entropy, global_step=global_step)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))


# 启动session
sess = tf.Session()
sess.run([tf.global_variables_initializer()])
tf.train.start_queue_runners(sess=sess)
old_time = time.time()
for i in range(15000):
    image_batch, label_batch = sess.run([images_train, labels_train])
    # one-hot编码
    label_b = np.eye(10, dtype=float)[label_batch]

    train_step.run(feed_dict={x:image_batch, y:label_b, train:1}, session=sess)

    if i % 200 == 0:
        new_time = time.time()
        cost_time = new_time-old_time
        old_time = new_time
        train_accuracy = accuracy.eval(feed_dict={x:image_batch, y:label_b}, session=sess)
        print('After %d training step, training accuracy:%g' % (i, train_accuracy))
        print('cost time is :%.2f s' % cost_time)

# 放入测试数据
image_batch, label_batch = sess.run([images_test, labels_test])
label_b = np.eye(10, dtype=float)[label_batch]
print('Finished! Test accuracy %g' % accuracy.eval(feed_dict={x:image_batch, y:label_b}, session=sess))
