# -*-coding:utf-8 -*-
# 损失函数loss:预测值(y)与已知答案(y_)的差距
# loss_mse = tf.reduce_mean(tf.square(y_-y))
# 交叉熵ce(Cross Entorpy):表示两个概率分布之间的距离
# H(y_, y) = -∑ y_ * logy


import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
seed = 23455   # 引入随机数种子是为了方便生成一致的数据

rdm = np.random.RandomState(seed)
X = rdm.rand(32, 2)
Y_ = [[x1 + x2 + (rdm.rand()/10.0 - 0.05)] for (x1, x2) in X]

# 定义神经网络的输入，参数和输出，定义向前传播过程
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# 定义损失函数及反向传播方法
# 定义损失函数为MSE,反向传播方法为梯度下降
loss_mse = tf.reduce_mean(tf.square(y_ - y))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)

# 生成回话，训练steps轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 2000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 32
        end = (i * BATCH_SIZE) % 32 + BATCH_SIZE
        sess.run(train_step, feed_dict={x:X[start:end], y_:Y_[start:end]})
        if i % 500 == 0:
            print('After %d training steps:, w1 is ' % (i))
            print(sess.run(w1),'\n')
    print('Final w1 is:\n', sess.run(w1))
