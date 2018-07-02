# -*-coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 定义ip和端口
strps_hosts = 'localhost:1681'
strworker_hosts = 'localhost:1682, localhost:1683'

# 定义角色名称
strjob_name = 'ps'
task_index = 0

# 将字符串转成数组
ps_hosts = strps_hosts.split(',')
worker_hosts = strworker_hosts.split(',')
cluster_spec = tf.train.ClusterSpec({'ps':ps_hosts, 'worker':worker_hosts})

# 创建server
server = tf.train.Server(
    {'ps':ps_hosts, 'worker':worker_hosts},
    job_name = strjob_name,
    task_index = task_index
)

# ps角色使用join进行等待
if strjob_name == 'ps':
    print('wait')
    server.join()

plotdata = {'batchsize':[], 'loss':[]}
def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w  else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

with tf.device(tf.train.replica_device_setter(
        worker_device='/job:worker/task:%d' % task_index,
    cluster=cluster_spec)):


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

    # 获得迭代次数
    global_step = tf.train.get_or_create_global_step()

    # 前向结构
    # tf.multiply()函数。两个数相乘
    z = tf.multiply(X, W) + b
    # 将预测中以直方图的形式显示
    tf.summary.histogram('z', z)

    # 反向优化
    # 定义一个cost，等于生成值与真实值的平方差
    cost = tf.reduce_mean(tf.square(Y - z))
    # 将损失以标量形式显示
    tf.summary.scalar('loss_fnction', cost)

    # 定义一个学习率，代表调整参数的速度
    # 值越大，表示调整的速度越快，但不精确；值越小，表示调整的速度越慢，但精度高
    learning_rate = 0.01
    # GradientDescentOptimizer()。是一个封装好的梯度下降函数
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step)

    saver = tf.train.Saver(max_to_keep=1)

    # 初始化前面所有的变量，如果后面再有变量，则不会再初始化
    init = tf.global_variables_initializer()

# 定义参数
# 迭代次数
training_epochs = 2200
display_step = 2

# 创建Supervisor,管理session
sv = tf.train.Supervisor(is_chief=(task_index==0),
                         logdir='log/super/',
                         init_op=init,
                         summary_op=None,
                         saver=saver,
                         global_step=global_step,
                         save_model_secs=5
                         )


print('sess OK')
# 启动session
with sv.managed_session(server.target) as sess:
    print(global_step.eval(sessioin=sess))

    # 存放批次值和损失值
    for epoch in range(global_step.eval(session=sess), training_epochs*len(train_X)):
        for(x, y) in zip(train_X, train_Y):
            _, epochs = sess.run([optimizer, global_step], feed_dict={X:x, Y:y})

            # 生成summary
            summary_str = sess.run(merged_summary_op, feed_dict={X:x, Y:y})
            sv.summary_computerd(sess, summary_str, global_step=epoch)

            # 显示训练中的详细信息
            if epoch % display_step == 0:
                loss = sess.run(cost, feed_dict={X:train_X, Y:train_Y})
                print("Epoch:", epoch+1, "cost=", loss, "W=", sess.run(W), "b=", sess.run(b))
                if not (loss == "NA"):
                    plotdata["batchsize"].append(epoch)
                    plotdata["loss"].append(loss)
    print("Finished!")
    sv.saver.save(sess, 'log, mnist_with_summaries/' + 'sv.cpk', global_step=epoch)
sv.stop()

