# -*- coding: utf-8 -*-

# LSTM-Regression

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# define the super parameters
BATCH_START = 0     # 建立 batch data 时的起点 index
TIME_STEPS = 20     # 用于 backpropagation through time 的 time_steps
BATCH_SIZE = 50     # batch size
INPUT_SIZE = 1      # sin 函数的输入 size
OUTPUT_SIZE = 1     # cos 函数的输出 size
CELL_SIZE = 10      # hidden unit size of RNN
LR = 0.006          # learning rate

# get batch of data
def get_batch():
    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    xs = np.arange(BATCH_START, BATCH_START + TIME_STEPS * BATCH_SIZE)
    xs = xs.reshape((BATCH_SIZE, TIME_STEPS))
    xs = xs / (10 * np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS  # update the start index batch
    # return seq, res, xs: shape (batch, step, input)
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]

# define LSTM-RNN
class LSTMRNN(object):
    # n_step: 一个batch的time_steps
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)
            
    def add_input_layer(self):
        # transform to the shape: (batch*n_steps, input_size)
        layer_in_x = tf.reshape(self.xs, shape=[-1, self.input_size], name='2_2D')
        # Ws_in shape: (input_size, cell_size)
        Ws_in = self._weigth_variable(shape=[self.input_size, self.cell_size])
        # bs_in shape: (cell_size,)
        bs_in = self._bias_variable(shape=[self.cell_size,])
        # layer_in_y: output of the input layer (y), shape: (batch*n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            layer_in_y = tf.matmul(layer_in_x, Ws_in) + bs_in
        # reshape layer_in_y -> (batch, n_steps, cell_size)
        self.layer_in_y = tf.reshape(layer_in_y, shape=[self.batch_size, self.n_steps, self.cell_size], name='2_3D')

    def add_cell(self):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.cell_size, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
                cell=lstm_cell, inputs=self.layer_in_y, initial_state=self.cell_init_state, time_major=False)

    def add_output_layer(self):
        # shape: (batch*n_steps, cell_size)
        layer_out_x = tf.reshape(self.cell_outputs, shape=[-1, self.cell_size], name='2_2D')
        # Ws_out shape: (cell_size, input_size)
        Ws_out = self._weigth_variable(shape=[self.cell_size, self.output_size])
        # bs_out shape: (output_size,)
        bs_out = self._bias_variable(shape=[self.output_size,])
        # layer_out_y shape: (batch*n_steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(layer_out_x, Ws_out) + bs_out

    def compute_cost(self):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                logits=[tf.reshape(self.pred, [-1], name='reshape_pred')],
                targets=[tf.reshape(self.ys, [-1], name='reshape_target')],
                weights=[tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
                average_across_timesteps=True,
                softmax_loss_function=self.squared_error,
                name='losses'
                )
        with tf.name_scope('average_cost'):
            self.cost = tf.div(tf.reduce_sum(losses, name='losses_sum'), self.batch_size,
                               name='average_cost')
            tf.summary.scalar(name='cost', tensor=self.cost)
    
    def squared_error(self, labels, logits):
        return tf.square(tf.subtract(labels, logits))
    
    def _weigth_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0.0, stddev=1.0)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)
    
    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(value=0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)


if __name__ == '__main__':
    # create LSTM-RNN model
    model = LSTMRNN(n_steps=TIME_STEPS, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, cell_size=CELL_SIZE, batch_size=BATCH_SIZE)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(logdir='D:/logs', graph=sess.graph)
    
    sess.run(tf.global_variables_initializer())
    
    plt.ion()   # continuous display
    plt.show()
    
    for i in range(201):
        seq, res, xs = get_batch()
        if i == 0:
            feed_dict = {
                    model.xs: seq,
                    model.ys: res
                    # automatically create initial state
                    }
        else:
            feed_dict = {
                    model.xs: seq,
                    model.ys: res,
                    model.cell_init_state: state
                    }
        _, cost, state, pred = sess.run(
            [model.train_op, model.cost, model.cell_final_state, model.pred],
            feed_dict=feed_dict)
        
        # plot the result
        plt.plot(xs[0,:], res[0].flatten(), 'r', xs[0,:], pred.flatten()[:TIME_STEPS], 'b--')
        plt.ylim((-1.2, 1.2))
        plt.draw()
        plt.pause(0.3)

        # print the cost
        if i % 20 == 0:
            print('step %3d, cost: %8.4f' % (i, cost))
            result = sess.run(merged, feed_dict=feed_dict)
            writer.add_summary(result, i)
    
    sess.close()
