# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
import random
import collections
from collections import Counter
import jieba

from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['font.family']='STSong'
mpl.rcParams['font.size']=20

training_file = 'wordstest.txt'


# 中文字
def get_ch_label(txt_file):
    labels = ''
    with open(txt_file, 'rb') as f:
        for label in f:
            labels += labels + label.decode('utf-8')
    return labels


# 分词
def fenci(training_data):
    seg_list = jieba.cut(training_data)
    training_ci = ' '.join(seg_list)
    training_ci = training_ci.split()
    training_ci = np.array(training_ci)
    training_ci = np.reshape(training_ci, [-1, ])
    return training_ci


def bulid_dataset(words, n_words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


# 获取批次数据
def generate_batch(data, batch_size, num_skips, skip_window):

    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)

    if data_index + span > len(data):
        data_index = 0

    buffer.extend(data[data_index:data_index + span])
    data_index += span

    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)

            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        if data_index == len(data):
            buffer = data[:span]
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


# 可视化
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >=len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y),  xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.savefig(filename)


training_data = get_ch_label(training_file)
print('总字数', len(training_data))
training_ci = fenci(training_data)
print('总词数', len(training_ci))
training_label, count, dictionary, words = bulid_dataset(training_ci, 350)

word_size = len(dictionary)
print('字典次数', word_size)

print('Sample data', training_label[:10], [words[i] for i in training_label[:10]])

data_index = 0

batch, labels = generate_batch(training_label, batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], words[batch[i]], '->', labels[i, 0], words[labels[i, 0]])


# 定义取样参数

batch_size = 128
# embeding vector 的维度
embedding_size = 128
# 左右词数量
skip_window = 1
# 一个input生成2个标签
num_skips = 2

valid_size = 16
# 取样数据的分布范围
valid_window = int(word_size/2)
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
# 负采样的个数
num_sampled = 64


# 定义模型变量

tf.reset_default_graph()

train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# 在cpu上执行
with tf.device('/cpu:0'):
    # 查找embedings
    embeddings = tf.Variable(tf.random_uniform([word_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # 计算NCE的loss值
    nce_weights = tf.Variable(tf.truncated_normal(
        [word_size, embedding_size],
        stddev=1.0/tf.sqrt(np.float32(embedding_size))))
    nce_biases = tf.Variable(tf.zeros([word_size]))


# 定义损失函数和优化器

loss = tf.reduce_mean(
    tf.nn.nce_loss(weights=nce_weights,
                   biases=nce_biases,
                   labels=train_labels,
                   inputs=embed,
                   num_sampled=num_sampled,
                   num_classes=word_size
                   )
)
optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

# 计算minibatch examples 和所有embedings的cosine相似度
norm = tf.sqrt(tf.reduce_mean(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)


# 训练模型

num_steps = 100001
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('Initialized')

    avg_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(training_label, batch_size, num_skips, skip_window)
        feed_dict = {train_inputs:batch_inputs, train_labels:batch_labels}
        _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
        avg_loss += loss_val

        # emv = sess.run(embed, feed_dict={train_inputs: [37,18]})
        # print('emv-------', emv[0])

        if step % 2000 == 0:
            if step > 0:
                avg_loss /= 2000
            print('Average loss at step:', step, 'avg_loss', avg_loss)
            average_loss = 0


# 输入验证数据，显示效果

        if step % 10000 == 0:
            sim = similarity.eval(session=sess)
            for i in range(valid_size):
                valid_word = words[valid_examples[i]]
                # 取排名最靠前的8个词
                top_k = 8
                # argsort函数返回的是数组值从小到大的索引值
                nearest = (-sim[i, :]).argsort()[:top_k + 1]
                log_str = 'Nearest
                to %s'  % valid_word
                for k in range(top_k):
                    close_word = words[nearest[k]]
                    log_str = '%s, %s' % (log_str, close_word)
                print(log_str)


# 词向量可视化

    final_embeddings = normalized_embeddings.eval()

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 80
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
labels = [words[i] for i in range(plot_only)]

plot_with_labels(low_dim_embs, labels)
