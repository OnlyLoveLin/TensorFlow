# -*- coding:utf-8 -*-

import tensorflow as tf

labels = [[0,0,1],[0,1,0]]
logits = [[2,0.5,6],[0.1,0,3]]

logits_scaled = tf.nn.softmax(logits)
logits_scaled2 = tf.nn.softmax(logits_scaled)

result1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
result2 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits_scaled)
result3 = -tf.reduce_sum(labels*tf.log(logits_scaled), 1)

with tf.Session() as sess:
    print('sclaed=' ,sess.run(logits_scaled))
    print('sclaed2=', sess.run(logits_scaled2)) # 经过第二次的softmax后，分布概率会有变化
    print('rel1=', sess.run(result1), '\n') # 正确的方式
    print('rel2=', sess.run(result2), '\n')
    print('rel3=', sess.run(result3), '\n')
