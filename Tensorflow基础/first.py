# -*-coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# np.linspace()函数。在指定的间隔内返回均匀间隔的数字
train_X = np.linspace(-1, 1, 100)
# Y的值。　
# np.random.randn(*train_X.shape) 等价与np.random.randn(100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3

# x和y为占位符，一个代表x的输入;另一个代表对应的真实值y
X = tf.placeholder('float')
Y = tf.placeholder('float')

# 模型参数。w被初始化程[-1,1]的随机数,b的初始化为0,都是一维的数字
W = tf.Variable(tf.random_normal([1]), name='weight')

b = tf.Variable(tf.zeros([1]), name='bias')

# 前向结构
# tf.multiply()函数。两个数相乘
z = tf.multiply(X, W) + b

# 反向优化
# 定义一个cost，等于生成值与真实值的平方差
cost = tf.reduce_mean(tf.square(Y - z))
# 定义一个学习率，代表调整参数的速度
# 值越大，表示调整的速度越快，但不精确；值越小，表示调整的速度越慢，但精度高
learning_rate = 0.01
# GradientDescentOptimizer()。是一个封装好的梯度下降函数
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 初始化所有变量
init = tf.global_variables_initializer()
# 定义参数
training_epochs = 20
display_step = 2

# 启动session
with tf.Session() as sess:
    sess.run(init)
    # 存放批次值和损失值
    plotdata = {"batchsize":[], "loss":[]}
    for epoch in range(training_epochs):
        for(x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X:x, Y:y})

        # 显示训练中的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X:train_X, Y:train_Y})
            print("Epoch:", epoch+1, "cost=", loss, "W=", sess.run(W), "b=", sess.run(b))
            if not (loss == "NA"):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)
    print("Finished!")
    print("cost=", sess.run(cost, feed_dict={X:train_X, Y:train_Y}), "w=", sess.run(W), "b=", sess.run(b))


    plotdata = {'batchsize':[], 'loss':[]}
    def moving_average(a, w=10):
        if len(a) < w:
            return a[:]
        return [val if idx < w  else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fittedline')
    plt.legend()
    plt.show()

    plotdata['avgloss'] = moving_average(plotdata['loss'])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata['batchsize'], plotdata['avgloss'], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minbatch run vs. Training loss')
    plt.show()

    print('x=0.2, z=', sess.run(z, feed_dict={X:0.2}))
