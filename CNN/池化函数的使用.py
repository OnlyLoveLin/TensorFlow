# -*-coding:utf-8 -*-

import tensorflow as tf


# 定义输入数据
img = tf.constant([
    [[0.0,4.0], [0.0,4.0], [0.0,4.0], [0.0,4.0]],
    [[1.0,5.0], [1.0,5.0], [1.0,5.0], [1.0,5.0]],
    [[2.0,6.0], [2.0,6.0], [2.0,6.0], [2.0,6.0]],
    [[3.0,7.0], [3.0,7.0], [3.0,7.0], [3.0,7.0]]
])

img = tf.reshape(img, [1,4,4,2])


# 定义池化操作
# 该操作是常用的，一般步长都会设成与池化滤波器尺寸一致
pooling = tf.nn.max_pool(img, [1,2,2,1], [1,2,2,1], padding='VALID')

pooling1 = tf.nn.max_pool(img, [1,2,2,1], [1,1,1,1], padding='VALID')

pooling2 = tf.nn.avg_pool(img, [1,4,4,1], [1,1,1,1], padding='SAME')

# 全局池化，一般放在最后一层，用于表达图像通过卷积网络处理后的最终特征
pooling3 = tf.nn.avg_pool(img, [1,4,4,1], [1,4,4,1], padding='SAME')

# 将数据转置
nt_hpool2_flat = tf.reshape(tf.transpose(img), [-1,16])
# 将数据转置后的均值操作
pooling4 = tf.reduce_mean(nt_hpool2_flat, 1)


# 运行池化操作
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print('images:\n', sess.run(img))
    print('pooling:\n', sess.run(pooling))
    print('pooling1:\n', sess.run(pooling1))
    print('pooling2:\n', sess.run(pooling2))
    print('pooling3:\n', sess.run(pooling3))
    print('pooling4:\n', sess.run(pooling4))
    print('flat:\n', sess.run(nt_hpool2_flat))

