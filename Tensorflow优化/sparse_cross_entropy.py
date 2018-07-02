# -*- coding:utf-8 -*-

import tensorflow as tf

# sparse标签
# 表明labels中共分为3个类，[2,1]等价与[0,0,1]与[0,1,0]
labels = [2,1]
logits = [[2,0.5,6],[0.1,0,3]]


result5 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

with tf.Session() as sess:
    print('rel5=', sess.run(result5), '\n') # 正确的方式
