# -*- coding: utf-8 -*-

# solve the overfitting by using dropout

import tensorflow as tf
from sklearn import datasets, preprocessing, cross_validation

# load data
digits = datasets.load_digits()
X = digits.data
y = digits.target
y = preprocessing.LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)

def nn_layer(layername, input_tensor, in_dim, out_dim, act=None):
    with tf.name_scope(layername):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.random_normal([in_dim, out_dim]), name='W')
            tf.summary.histogram(layername+'/weights', weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_dim]) + 0.1, name='b')
            tf.summary.histogram(layername+'/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            z = tf.matmul(input_tensor, weights) + biases
            z = tf.nn.dropout(z, keep_prob)
        if act is None:
            outputs = z
        else:
            outputs = act(z)
        tf.summary.histogram(layername+'/outputs', outputs)
        return outputs

# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64]) # 8*8
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
hidden1 = nn_layer("layer_1", xs, 64, 50, act=tf.nn.tanh)
y_pred = nn_layer("layer_2", hidden1, 50, 10, act=tf.nn.softmax)

# the loss between prediction value and true value
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(y_pred), reduction_indices=[1]))
    tf.summary.scalar('loss', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

# start session
init = tf.global_variables_initializer()
sess = tf.Session()

# tensor board
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('D:/logs/train', sess.graph)
test_writer = tf.summary.FileWriter('D:/logs/test', sess.graph)

sess.run(init)

for i in range(5000):
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob:0.5})
    if i % 50 == 0:
        # record loss
        train_res = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob:1})
        test_res = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob:1})
        train_writer.add_summary(train_res, i)
        test_writer.add_summary(test_res, i)

train_writer.close()
test_writer.close()
sess.close()
