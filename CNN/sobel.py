# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf


# 读取图片
myimg = mpimg.imread('3.jpg')
# 显示图片
plt.imshow(myimg)
# 不显示坐标轴
plt.axis('off')
plt.show()
print(myimg.shape)
img_height = myimg.shape[0]
img_width = myimg.shape[1]

full = np.reshape(myimg, [1, img_height, img_width, 3])
inputfull = tf.Variable(tf.constant(1.0, shape=[1, img_height, img_width, 3]))

# 定义卷积核
filter = tf.Variable(tf.constant([[-1.0,-1.0,-1.0], [0,0,0], [1.0,1.0,1.0],
                                  [-2.0,-2.0,-2.0], [0,0,0], [2.0,2.0,2.0],
                                  [-1.0,-1.0,-1.0], [0,0,0], [1.0,1.0,1.0]
                                  ], shape=[3,3,3,1]))

# 卷积操作
op = tf.nn.conv2d(inputfull, filter, strides=[1,1,1,1], padding='SAME')
# 对数据进行归一化处理
o =tf.cast(((op-tf.reduce_min(op))/(tf.reduce_max(op)-tf.reduce_min(op)))*255, tf.uint8)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    t, f = sess.run([o, filter], feed_dict={inputfull:full})

    t = np.reshape(t, [img_height, img_width])
    plt.imshow(t, cmap='Greys_r')
    plt.axis('off')
    plt.show()


