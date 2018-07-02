# -*-coding:utf-8 -*-

import tensorflow as tf


tf.reset_default_graph()
global_step = tf.train.get_or_create_global_stp())
step = tf.assign_add(global_step, 1)
# 设置检查点路径为log/checkpoints
with tf.train.MonitoredTrainingSession(checkpoint_dir='log/checkpoints', save_checkpoint_secs=2) as sess:
    print(sess.run([global_step]))
    while not sess.should_stop():
        # 启用死循环，当sess不结束是就不停止
        i = sess.run(step)
        print(i)
