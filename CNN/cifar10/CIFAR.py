# -*- coding:utf-8 -*-

import tensorflow as tf
import pylab
import cifar10_input


# 取数据
batch_size = 128
data_dir = 'cifar-10-batches-bin/'
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

sess = tf.InteractiveSession()
tf.global_variables_initializer()
# 运行队列
tf.train.start_queue_runners()
image_batch, label_batch = sess.run([images_test, labels_test])
# 输出图像数据和标签数据
print('__\n', image_batch[0])
print('__\n', label_batch[0])
pylab.imshow(image_batch[0])
pylab.show()
