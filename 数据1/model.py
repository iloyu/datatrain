#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 19:40:37 2018

@author: wanghao
"""
import tensorflow as tf
import numpy as np

def regression(inputs):
    #batchsize = 4
    #inputs = tf.reshape(inputs, [batchsize,165])
    with tf.variable_scope("regression") as scope:
#        w1 = tf.Variable(tf.random_normal([165,1024],stddev = 0.01))
#        w2 = tf.Variable(tf.random_normal([1024,165],stddev = 0.01))
#
#        b1 = tf.Variable(tf.zeros([1024]) + 0.1)
#        b2 = tf.Variable(tf.zeros([165]) + 0.1)
#
#        out1 = tf.nn.relu(tf.matmul(inputs, w1) + b1)
#
#        out2 = tf.matmul(out1, w2) + b2
#        return out2

#        w1 = tf.get_variable('w1',shape=[166,512], initializer =tf.contrib.layers.xavier_initializer() , regularizer = tf.contrib.layers.l2_regularizer(0.5))#tf.contrib.layers.xavier_initializer
#        w2 = tf.get_variable('w2',shape=[512,1024], initializer = tf.contrib.layers.xavier_initializer() , regularizer = tf.contrib.layers.l2_regularizer(0.5))#tf.random_normal_initializer(mean=0,stddev=0.01)
#        w3 = tf.get_variable('w3',shape=[1024,1024], initializer = tf.contrib.layers.xavier_initializer() , regularizer = tf.contrib.layers.l2_regularizer(0.5))
#        w4 = tf.get_variable('w4',shape=[1024,512], initializer = tf.contrib.layers.xavier_initializer() , regularizer = tf.contrib.layers.l2_regularizer(0.5))
#        w5 = tf.get_variable('w5',shape=[512,165], initializer = tf.contrib.layers.xavier_initializer() , regularizer = tf.contrib.layers.l2_regularizer(0.5))
#        #tf.get_variable(name,  shape, initializer): name就是变量的名称，shape是变量的维度，initializer是变量初始化的方式，
#
#        b1 = tf.get_variable('b1', shape=[512], initializer=tf.zeros_initializer())
#        b2 = tf.get_variable('b2', shape=[1024], initializer=tf.zeros_initializer())
#        b3 = tf.get_variable('b3', shape=[1024], initializer=tf.zeros_initializer())
#        b4 = tf.get_variable('b4', shape=[512], initializer=tf.zeros_initializer())
#        b5 = tf.get_variable('b5', shape=[165], initializer=tf.zeros_initializer())
#
#        out1 = tf.nn.relu(tf.matmul(inputs,w1) + b1)
#        out2 = tf.nn.relu(tf.matmul(out1,w2) + b2)
#        out3 = tf.nn.relu(tf.matmul(out2,w3) + b3)
#        out4 = tf.nn.relu(tf.matmul(out3,w4) + b4)
#        out5 = tf.matmul(out4,w5) + b5
#        return out5

        w1 = tf.get_variable('w1',shape=[166,1024], initializer =tf.contrib.layers.xavier_initializer() , regularizer = tf.contrib.layers.l1_regularizer(0.01))#tf.contrib.layers.xavier_initializer
        w2 = tf.get_variable('w2',shape=[1024,1024], initializer = tf.contrib.layers.xavier_initializer() , regularizer = tf.contrib.layers.l1_regularizer(0.01))
        w3 = tf.get_variable('w3',shape=[1024,165], initializer = tf.contrib.layers.xavier_initializer() , regularizer = tf.contrib.layers.l1_regularizer(0.01))
         #tf.get_variable(name,  shape, initializer): name就是变量的名称，shape是变量的维度，initializer是变量初始化的方式，

        b1 = tf.get_variable('b1', shape=[1024], initializer=tf.zeros_initializer())
        b2 = tf.get_variable('b2', shape=[1024], initializer=tf.zeros_initializer())
        b3 = tf.get_variable('b3', shape=[165], initializer=tf.zeros_initializer())

        out1 = tf.nn.relu(tf.matmul(inputs, w1) + b1)
        out2 = tf.nn.relu(tf.matmul(out1,w2) + b2)
        out3 = tf.matmul(out2,w3) + b3
        return out3



#        w1 = tf.Variable(tf.random_normal([165,256],stddev = 0.01))
#        w2 = tf.Variable(tf.random_normal([256,512],stddev = 0.01))
#        w3 = tf.Variable(tf.random_normal([512,1024],stddev = 0.01))
#        w4 = tf.Variable(tf.random_normal([1024,512],stddev = 0.01))
#        w5 = tf.Variable(tf.random_normal([512,256],stddev = 0.01))
#        w6 = tf.Variable(tf.random_normal([256,165],stddev = 0.01))
#
#        b1 = tf.Variable(tf.zeros([256]) + 0.1)
#        b2 = tf.Variable(tf.zeros([512]) + 0.1)
#        b3 = tf.Variable(tf.zeros([1024]) + 0.1)
#        b4 = tf.Variable(tf.zeros([512]) + 0.1)
#        b5 = tf.Variable(tf.zeros([256]) + 0.1)
#        b6 = tf.Variable(tf.zeros([165]) + 0.1)
#
#        out1 = tf.nn.relu(tf.matmul(inputs, w1) + b1)
#        out2 = tf.nn.relu(tf.matmul(out1, w2) + b2)
#        out3 = tf.nn.relu(tf.matmul(out2, w3) + b3)
#        out4 = tf.nn.relu(tf.matmul(out3, w4) + b4)
#        out5 = tf.nn.relu(tf.matmul(out4, w5) + b5)
#        out6 = tf.matmul(out5, w6) + b6
#        return out6

