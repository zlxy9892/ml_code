# -*- coding: utf-8 -*-

# placeholder
import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    oo = sess.run(output, feed_dict={input1: [2.], input2: [4.]})
    print(oo)