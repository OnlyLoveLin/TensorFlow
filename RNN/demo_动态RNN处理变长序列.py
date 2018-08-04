# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

tf.reset_default_graph()

# 创建输入数据
X = np.random.randn(2, 4, 5)

# 第二个样本长度为3
X[1, 1: ] = 0
seq_lengths = [4, 1]

# 分别建立一个lstm和GRU的cell,比较输出状态
cell = tf.contrib.rnn.BasicLSTMCell(num_units=3, state_is_tuple=True)
gru = tf.contrib.rnn.GRUCell(3)

# 如果没有initial_state,必须指定dtype
outputs, last_states = tf.nn.dynamic_rnn(cell, X, seq_lengths, dtype=tf.float64)
gruoutputs, grulast_states = tf.nn.dynamic_rnn(gru, X, seq_lengths, dtype=tf.float64)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
result, sta, gruout, grusta = sess.run([outputs, last_states, gruoutputs, grulast_states])

print('全序列:\n', result[0])

print('LSTM短序列:\n', result[1])
print('LSTM的状态：',len(sta), '\n', sta[1])

print('GRU的短序列：\n',gruout[1])
print('GRU的状态：', len(grusta), '\n', grusta[1])
