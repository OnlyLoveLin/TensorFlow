# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# MNSIT Data的输入
n_input = 28
# 序列的长度
n_steps = 28
# 隐藏层的个数
n_hidden = 128
# MNIST 分类的个数
n_classes = 10

tf.reset_default_graph()

# 定义占位符
x = tf.placeholder('float', [None, n_steps, n_input])
y = tf.placeholder('float', [None, n_classes])

x1 = tf.unstack(x, n_steps, 1)
# 前向cell
lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

# 反向cell
lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

# 创建动态RNN
outputs, outputs_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
# 因为前向和后向的输出结果是分开的，要将输出结果进行融合
outputs = tf.concat(outputs, 2)
# 进行转置
outputs  = tf.transpose(outputs, [1,0,2])
pred = tf.contrib.layers.fully_connected(outputs[-1], n_classes, activation_fn=None)

learning_rate = 0.001
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 评估模型节点
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

training_iters = 100000
batch_size = 128
display_step = 10

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    step = 1

    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))
        sess.run(optimizer, feed_dict={x:batch_xs, y:batch_ys})

        if step % display_step == 0:
            # 计算批次数据的准确率
            acc = sess.run(accuracy, feed_dict={x:batch_xs, y:batch_ys})
            # 计算批次损失率
            loss = sess.run(cost, feed_dict={x:batch_xs, y:batch_ys})
            print('Iter' + str(step*batch_size) + ', Minibatch Loss=' + '{:.6f}'.format(loss) + ', Training Accuracy=' + '{:.6f}'.format(acc))

        step += 1
        print('finish!')

        # 计算准确率
        test_len = 128
        test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
        test_label = mnist.test.labels[:test_len]
        print('Test Accuracy:', sess.run(accuracy, feed_dict={x:test_data, y:test_label}))



