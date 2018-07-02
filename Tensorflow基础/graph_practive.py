import numpy as np
import tensorflow as tf

c = tf.constant(0.0)

g = tf.Graph()
with g.as_default():
    c1 = tf.constant(0.1)
    print(c1.graph)
    print(g)
    print(c.graph)
    g2 = tf.get_default_graph()
    print(g2)

tf.reset_default_graph()
g3 = tf.get_default_graph()
print(g3)


