# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

## save the session ###  
#v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
#v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
#result = v1 + v2
#  
#saver = tf.train.Saver()
#
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    saver.save(sess, "Model/model.ckpt")


### restore the sess ###
# need to redefine the same shape and same dtype for the stored variables
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")  
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")  
result = v1 + v2  
  
saver = tf.train.Saver()
  
with tf.Session() as sess:  
    saver.restore(sess, "./Model/model.ckpt") # 注意此处路径前添加"./"  
    print(sess.run(result)) # [ 3.]  

import os
os.system('pause')