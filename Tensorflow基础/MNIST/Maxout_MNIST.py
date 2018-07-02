# -*- coding:utf-8 -*-
# 实验环境:Ubuntu python3.5
# 所需的库tensorflow

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pylab


mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# 重置图
tf.reset_default_graph()

# 定义占位符
# 数据集的维度是28*28=874
x = tf.placeholder(tf.float32, [None, 784])
# 共10个类别
y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.random_normal(([784, 10])))
b = tf.Variable(tf.zeros([10]))
z = tf.matmul(x, W) + b
maxout = tf.reduce_max(z, axis=1, keep_dims=True)
# 设置学习参数
W2 = tf.Variable(tf.truncated_normal([1, 10], stddev=0.1))
b2 = tf.Variable(tf.zeros([1]))

# 正向传播
# softmax分类
pred = tf.nn.softmax(tf.matmul(x, W) + b)

# 反向结构
# 损失函数
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# 定义参数
learning_rate = 0.04
# 使用梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


training_epochs = 100
batch_size = 100
display_step = 1

saver = tf.train.Saver()
model_path = 'log/521model.ckpt'

# 启动session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 启动循环开始训练
    for epoch in range(training_epochs):
        avg_cost = 0

        # total_batcj=550
        total_batch = int(mnist.train.num_examples/batch_size)
        # 循环所有的数据集
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 运行优化器
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})

            # 计算平均的loss值
            avg_cost += c / total_batch

        # 显示训练中的详细信息
        if (epoch+1) % display_step == 0:
            print('Epochs:', '%04d' % (epoch+1), 'cost:', '{:.9f}'.format(avg_cost))
    print('Finished!')

    # 测试模型
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算精确度
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accurary:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    # 保存模型
    save_path = saver.save(sess, model_path)
    print('Model saved in file: %s' % save_path)

# 读取模型
print('Starting loading model')
with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())
    # 恢复模型变量
    saver.restore(sess, model_path)

    # 测试模型
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))

    output = tf.argmax(pred, 1)
    batch_xs, batch_ys = mnist.train.next_batch(2)
    outputval, predv = sess.run([output, pred], feed_dict={x: batch_xs, y:batch_ys})
    print(outputval, predv, batch_ys)

    im = batch_xs[0]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()

    im = batch_xs[1]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()
