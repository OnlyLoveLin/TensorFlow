# -*-coding:utf-8 -*-
# 正则化缓解过拟合
# 正则化在损失函数中引入模型复杂度指标，利用给w加权值，弱化了训练数据的噪声（一般不正则化b）
# loss = loss(y与y_) + regularizer* loss(w)


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 30
seed = 2

# 基于seed产生随机数
rdm = np.random.RandomState(seed)
# 随机数返回300行2列的举证,表示300组坐标点(x0, x1)作为输入数据集
X = rdm.randn(300, 2)
# 从Ｘ这个300行2列的矩阵中取出一行，判断如果有两个坐标的平方和小于２，给Ｙ赋值为１，其余赋值为０
# 作为输入数据集的标签
Y_ = [int(x0*x0 + x1*x1 < 2) for (x0, x1) in X]

# 遍历Ｙ中的每个元素，１赋值‘red’其余赋值'blue'，这样可以可视化显示，人可以直接区分
Y_C = ['red' if y else 'blue' for y in Y_]

# 对数据集Ｘ和标签Ｙ进行shape整理，第一个元素为-1表示,随第二个参数计算得到；第二个元素表示多少列。
# 把Ｘ整理为ｎ行２列，把Ｙ整理为ｎ行１列
X = np.vstack(X).reshape(-1, 2)
Y_ = np.vstack(Y_).reshape(-1, 1)

print(X)
print(Y_)
print(Y_C)


# 用plt.scatter画出数据集Ｘ各行中第０行元素和第１列元素的点，即各行的(x0,
# x1),用各行Y_C对应的值表示颜色
plt.scatter(X[:,0], X[:,1], c = np.squeeze(Y_C))
plt.show()


# 定义神经网络的输入，参数和输出，定义前向传播过程
# 定义一个生成权重的函数
def get_weight(shape, regularizer):
    w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    # 把计算好w的正则化加在losses集合中
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

# 定义一个生成偏置值的函数

def get_bias(shape):
    b = tf.Variable(tf.constant(0.01, shape=shape))
    return b


# 定义占位符
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))


w1 = get_weight([2,11], 0.01)
b1 = get_bias([11])
y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = get_weight([11,1], 0.01)
b2 = get_bias([1])
y = tf.matmul(y1, w2) + b2  # 输出层没有激活函数

# 定义损失函数
# 均方误差
loss_mse = tf.reduce_mean(tf.square(y-y_))
# 正则化的损失函数。tf.add_n()可以把'losses'中的所有值相加
loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))


# 定义反向传播方法，不含正则化
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mse)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 40000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 300
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x:X[start:end], y_:Y_[start:end]})
        if i % 2000 == 0:
            loss_mse_v = sess.run(loss_mse, feed_dict={x:X, y_:Y_})
            print('After %d step, loss is: %f' %(i, loss_mse_v))

    # x在-3到3之间以步长为0.01，yy在-3到3之间以步长0.01，生成二维网格坐标点
    xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
    # 将xx, yy,拉直，并合并成一个2列的矩阵，得到一个网格坐标点的集合
    grid = np.c_[xx.ravel(), yy.ravel()]
    # 将网格坐标点为入神经网络，probs为输出
    probs = sess.run(y, feed_dict={x:grid})
    # 将probs的shape调整成xx的样子
    probs = probs.reshape(xx.shape)
    print('w1:\n', sess.run(w1))
    print('b1:\n', sess.run(b1))
    print('w2:\n', sess.run(w2))
    print('b2:\n', sess.run(b2))


plt.scatter(X[:,0], X[:,1],c=np.squeeze(Y_C))
plt.contour(xx, yy, probs, levels=[.5])
plt.show()


# 定义反向传播过程，包含正则化
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_total)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 40000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 300
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x:X[start:end], y_:Y_[start:end]})
        if i % 2000 == 0:
            loss_mse_v = sess.run(loss_mse, feed_dict={x:X, y_:Y_})
            print('After %d step, loss is: %f' %(i, loss_mse_v))

    # x在-3到3之间以步长为0.01，yy在-3到3之间以步长0.01，生成二维网格坐标点
    xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
    # 将xx, yy,拉直，并合并成一个2列的矩阵，得到一个网格坐标点的集合
    grid = np.c_[xx.ravel(), yy.ravel()]
    # 将网格坐标点为入神经网络，probs为输出
    probs = sess.run(y, feed_dict={x:grid})
    # 将probs的shape调整成xx的样子
    probs = probs.reshape(xx.shape)
    print('w1:\n', sess.run(w1))
    print('b1:\n', sess.run(b1))
    print('w2:\n', sess.run(w2))
    print('b2:\n', sess.run(b2))


plt.scatter(X[:,0], X[:,1],c=np.squeeze(Y_C))
plt.contour(xx, yy, probs, levels=[.5])
plt.show()


