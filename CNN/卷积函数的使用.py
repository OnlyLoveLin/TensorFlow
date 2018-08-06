# -*- coding:utf-8 -*-

import tensorflow as tf


# 定义输入变量
# 5x5尺寸，单通道
input1 = tf.Variable(tf.constant(1.0, shape=[1,5,5,1]))
# 5x5尺寸,双通道
input2 = tf.Variable(tf.constant(1.0, shape=[1,5,5,2]))
# 4x4尺寸，单通道
input3 = tf.Variable(tf.constant(1.0, shape=[1,4,4,1]))


# 定义卷积核变量
# 2x2尺寸，单通道，一个卷积核
filter1 = tf.Variable(tf.constant([-1.0,0,0,-1.0], shape=[2,2,1,1]))
# 2x2尺寸，单通道，两个卷积核
filter2 = tf.Variable(tf.constant([-1.0,0,0,-1.0,-1.0,0,0,-1.0], shape=[2,2,1,2]))
# 2x2尺寸，单通道，三个卷积核
filter3 = tf.Variable(tf.constant([-1.0,0,0,-1.0,-1.0,0,0,-1.0,-1.0,0,0,-1.0], shape=[2,2,1,3]))
# 2x2尺寸，双通道，四个卷积核
filter4 = tf.Variable(tf.constant([-1.0,0,0,-1.0,-1.0,0,0,-1.0,-1.0,0,0,-1.0,-1.0,0,0,-1.0], shape=[2,2,2,2]))
# 2x2尺寸，双通道，一个卷积核
filter5 = tf.Variable(tf.constant([-1.0,0,0,-1.0,-1.0,0,0,-1.0], shape=[2,2,2,1]))


# 定义卷积操作
op1 = tf.nn.conv2d(input1, filter1, strides=[1,2,2,1], padding='SAME')
op2 = tf.nn.conv2d(input1, filter2, strides=[1,2,2,1], padding='SAME')
op3 = tf.nn.conv2d(input1, filter3, strides=[1,2,2,1], padding='SAME')
op4 = tf.nn.conv2d(input2, filter4, strides=[1,2,2,1], padding='SAME')
op5 = tf.nn.conv2d(input2, filter5, strides=[1,2,2,1], padding='SAME')
op6 = tf.nn.conv2d(input3, filter1, strides=[1,2,2,1], padding='SAME')

vop1 = tf.nn.conv2d(input1, filter1, strides=[1,2,2,1], padding='VALID')
vop6 = tf.nn.conv2d(input3, filter2, strides=[1,2,2,1], padding='VALID')


# 运行输出
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 5x5单通道输入,生成1个feature map
    print('op1:\n', sess.run([op1, filter1]))
    print('--------------------------------')

    # 5x5单通道输入，生成2个feature map
    print('op2:\n', sess.run([op2, filter2]))
    print('--------------------------------')

    # 5x5单通道输入，生成3个feature map
    print('op3:\n', sess.run([op3, filter3]))
    print('--------------------------------')

    # 5x5双通道输入生成两个feature map
    print('op4:\n', sess.run([op4, filter4]))
    print('--------------------------------')

    # 5x5双通道输入，生成1个feature map
    print('op5:\n', sess.run([op5, filter5]))
    print('--------------------------------')

    # 4x4单通道输入，生成1个feature map
    print('op6:\n', sess.run([op6, filter1]))
    print('--------------------------------')

    # 5x5单通道输入，生成1个feature,不使用padding填充
    print('vop1:\n', sess.run([vop1, filter1]))
    print('--------------------------------')

    # 4x4单通道输入，生成1个feature,不使用padding填充
    print('vop6:\n', sess.run([vop6, filter1]))
    print('--------------------------------')

