# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mat1 = tf.constant([[3,3]])
mat2 = tf.constant([[2], [2]])
product = tf.matmul(mat1, mat2) # matrix multiply np.dot(mat1, mat2)

# method 1
#sess = tf.Session()
#result = sess.run(product)
#print(result)
#sess.close()

# method 2
with tf.Session() as sess:
    result = sess.run(product)
    print(result)

