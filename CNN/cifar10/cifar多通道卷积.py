# -*- coding:utf-8 -*-


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
def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1,6,6,1], strides=[1,6,6,1], padding='SAME')


# 定义占位符
x = tf.placeholder(tf.float32, [None, 24 , 24, 3])
y = tf.placeholder(tf.float32, [None, 10])

W_conv1 = weight_variable([5, 5, 3, 64])
b_conv1 = bias_variable([64])

x_image = tf.reshape(x, [-1, 24, 24, 3])

# 定义第一层卷积层
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# 定义第一层池化层
h_pool1 = max_pool_2x2(h_conv1)


W_conv2_5x5 = weight_variable([5, 5, 64, 64])
b_conv2_5x5 = bias_variable([64])

W_conv2_7x7 = weight_variable([7, 7, 64, 64])
b_conv2_7x7 = bias_variable([64])

W_conv2_3x3 = weight_variable([3, 3, 64, 64])
b_conv2_3x3 = bias_variable([64])

W_conv2_1x1 = weight_variable([1, 1, 64, 64])
b_conv2_1x1 = bias_variable([64])

h_conv2_5x5 = tf.nn.relu(conv2d(h_pool1, W_conv2_5x5) + b_conv2_5x5)
h_conv2_7x7 = tf.nn.relu(conv2d(h_pool1, W_conv2_7x7) + b_conv2_7x7)
h_conv2_3x3 = tf.nn.relu(conv2d(h_pool1, W_conv2_3x3) + b_conv2_3x3)
h_conv2_1x1 = tf.nn.relu(conv2d(h_pool1, W_conv2_1x1) + b_conv2_1x1)

h_conv2 = tf.concat([h_conv2_5x5, h_conv2_7x7, h_conv2_3x3, h_conv2_1x1], 3)

h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([5, 5, 256, 10])
b_conv3 = bias_variable([10])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

# 进行全局平均池化
nt_pool3 = avg_pool_2x2(h_conv3)

nt_pool3_flat = tf.reshape(nt_pool3, [-1, 10])

# 进行softmax分类
y_conv = tf.nn.softmax(nt_pool3_flat)

# 交叉熵
cross_entropy = -tf.reduce_sum(y*tf.log(y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

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

    train_step.run(feed_dict={x:image_batch, y:label_b}, session=sess)

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
