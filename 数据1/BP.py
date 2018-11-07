#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 17:54:53 2018

@author: iloyu
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from model import regression
import matplotlib.pyplot as plt
from numpy import *
tf.reset_default_graph()
Numsize=500#样本数
feature=165 #特征数
lrate=0.001 #学习速率
batch_size = 100#训练batch的大小
steps=20000#迭代次数
dimension=165#属性
channel=3 #x,y,z
total=400#训练样本数

train_start = 0
train_end = int(np.floor(0.8*Numsize))
test_start = train_end
test_end = Numsize

inputX= pd.read_csv('xDataR2.txt',header=None,sep=',')
inputY=pd.read_csv('yDataR2.txt',header=None,sep=',')

x_data =inputX.values#[:,0:dimension]
y_data=inputY.values#[:,0:dimension]

xdata_train = x_data[np.arange(train_start, train_end),:]
xdata_test = x_data[np.arange(test_start, test_end), :]
ydata_train =y_data[np.arange(train_start, train_end), :]
ydata_test = y_data[np.arange(test_start, test_end), :]

# print(xdata_test,xdata_test.shape)
# print(ydata_test,ydata_test.shape)
#归一化
xtrainMean=np.mean(xdata_train,axis=0)
xtrainStd=np.std(xdata_train,axis=0)
ytrainMean=np.mean(ydata_train,axis=0)
ytrainStd=np.std(ydata_train,axis=0)


#xtrainMax=np.max(xdata_train,axis=0)
#xtrainMin=np.min(xdata_train,axis=0)
#ytrainMax=np.max(ydata_train,axis=0)
#ytrainMin=np.min(ydata_train,axis=0)


#xdata_train-=xtrainMin
#xdata_train/=(xtrainMax-xtrainMin)
#
#xdata_test-=xtrainMin
#xdata_test/=(xtrainMax-xtrainMin)
#
#ydata_train-=ytrainMin
#ydata_train/=(ytrainMax-ytrainMin)
#
#ydata_test-=ytrainMin
#ydata_test/=(ytrainMax-ytrainMin)

xdata_train-=xtrainMean
xdata_train/=xtrainStd

xdata_test-=xtrainMean
xdata_test/=xtrainStd

ydata_train-=ytrainMean
ydata_train/=ytrainStd

ydata_test-=ytrainMean
ydata_test/=ytrainStd
#x channel
xf=[i for i in range(165) if i%3==0]
xdata_train1=xdata_train[:,xf]
ydata_train1=ydata_train[:,xf]
xdata_test1=xdata_test[:,xf]
ydata_test1=ydata_test[:,xf]

xtrainMean1=np.mean(xdata_train1,axis=0)
xtrainStd1=np.std(xdata_train1,axis=0)
ytrainMean1=np.mean(ydata_train1,axis=0)
ytrainStd1=np.std(ydata_train1,axis=0)

#y channel
xf=[i for i in range(165) if i%3==1]
xdata_train2=xdata_train[:,xf]
ydata_train2=ydata_train[:,xf]
xdata_test2=xdata_test[:,xf]
ydata_test2=ydata_test[:,xf]
#z channel
xf=[i for i in range(165) if i%3==2]
xdata_train3=xdata_train[:,xf]
ydata_train3=ydata_train[:,xf]
xdata_test3=xdata_test[:,xf]
ydata_test3=ydata_test[:,xf]
xdata_trainMix=np.concatenate((xdata_train1,xdata_train2,xdata_train3),1).reshape(400,3,55)
# print(xdata_train,ydata_train)
#转置
# xdata_train = np.transpose(x_data[np.arange(train_start, train_end), :])
# xdata_test = np.transpose(x_data[np.arange(test_start, test_end), :])
# ydata_train =np.transpose(y_data[np.arange(train_start, train_end), :])
# ydata_test = np.transpose(y_data[np.arange(test_start, test_end), :])

 # 2.定义节点准备接收数据
# global_step=tf.Variable(0)
# learning_rate=tf.train.exponential_decay(0.1,global_step,100,0.96,staircase=True
xs = tf.placeholder(tf.float32, [None,dimension],name="x_traindata")
ys = tf.placeholder(tf.float32, [None,dimension],name="y_traindata")
import tensorflow as tf
def LeakyRelu(x, leak=0.2, name="LeakyRelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * tf.abs(x)
#prediction = regression(xs)
def add_layer(wname,bname,inputs, in_size, out_size, activation_function=None):
    with tf.variable_scope("BPNet") as scope:
    # 构建权重 : in_size * out)_sieze 大小的矩阵
        weights = tf.get_variable(wname,shape=[in_size,out_size], initializer=tf.contrib.layers.variance_scaling_initializer(), regularizer = tf.contrib.layers.l1_regularizer(0.01))#xavier_initializer()
        # 构建偏置 : 1 * out_size 的矩阵
        b = tf.get_variable(bname, shape=[1,out_size], initializer=tf.zeros_initializer())
        # 矩阵相乘
        Wx_plus_b = tf.matmul(inputs, weights) + b
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return weights,outputs  # 得到输出数据
hidden_layers1=256
hidden_layers2=512
hidden_layers3=1024
hidden_layers4=512#1024
hidden_layers5=256
hidden_layers6=dimension
## 构建输入层到隐藏层,假设隐藏层有 hidden_layers 个神经元

w1,h1=add_layer('w1','b1',xs,dimension,hidden_layers1,activation_function=tf.nn.relu)
w2,h2=add_layer('w2','b2',h1,hidden_layers1,hidden_layers2,activation_function=tf.nn.relu)
w3,h3=add_layer('w3','b3',h2,hidden_layers2,hidden_layers3,activation_function=tf.nn.relu)
w4,h4=add_layer('w4','b4',h3,hidden_layers3,hidden_layers4,activation_function=tf.nn.relu)
w5,h5=add_layer('w5','b5',h4,hidden_layers4,hidden_layers5,activation_function=tf.nn.relu)
w6,prediction=add_layer('w6','b6',h5,hidden_layers5,hidden_layers6)
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h4, keep_prob)
#loss=tf.reduce_mean(tf.square(prediction-ys))
def acc(disTest):
    count=0
    for x in range(len(disTest)):
        for i in disTest[x]:
            if abs(i)<1:
                count+=1
    measure=count/np.size(disTest)
    return measure
loss=tf.reduce_mean(tf.reduce_sum(tf.squared_difference(prediction,ys),1))

#tf.reduce_sum(tf.sign(tf.abs(prediction-ys)-1))


train_step =tf.train.AdamOptimizer(lrate).minimize(loss)


#image_summary = tf.summary.image
scalar_summary = tf.summary.scalar
#histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter

loss_summary = scalar_summary('loss',loss)
validloss_summary = scalar_summary('validloss',loss)


#tf.train.GradientDescentOptimizer(lrate).minimize(loss) # SGD,随机梯度下降
# 使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放
#内存，所以会导致碎片
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
#print('将计算图写入事件文件,在TensorBoard里查看')
#writer = tf.summary.FileWriter(logdir='logs/8_2_BP', graph=tf.get_default_graph())
#writer.close()

savetrain = tf.train.Saver(max_to_keep = 1)

test={xs: xdata_test, ys: ydata_test}
testx={xs: xdata_test}
trainx={xs:xdata_train}
train={xs: xdata_train, ys: ydata_train}

#test={xs: xdata_test1, ys: ydata_test1}
#testx={xs: xdata_test1}
#trainx={xs:xdata_train1}
#train={xs: xdata_train1, ys: ydata_train1}
with tf.Session(config = config) as sess:
    # 初始化所有变量
    init = tf.global_variables_initializer()
    sess.run(init)
    trainmerged = merge_summary([loss_summary])
    validmerged = merge_summary([validloss_summary])
    writer = SummaryWriter('./logs',sess.graph)
    costTrain_history=[ ]
    costTest_history=[ ]
    acc_history=[ ]
    acc_Testhistory=[ ]
#    summary_writer_train = tf.summary.FileWriter(FLAGS.logs_dir + '/train', sess.graph)
#    summary_writer_val = tf.summary.FileWriter(FLAGS.logs_dir + '/val')
    for i in range(1,steps+1):
    #     batchSize使用
#         start=(i*batch_size)%total
#         end=min(start+batch_size,total)
#         if start==end:
#             start=398
#             end=399

         sess.run(train_step, feed_dict=train)

         costTrain=sess.run(loss, feed_dict=train)
         costTrain_history.append(costTrain)

         costTest=sess.run(loss, feed_dict=test)
         costTest_history.append(costTest)
         #调用sess.run运行图，生成一步的训练过程数据
         train_summary = sess.run(trainmerged,feed_dict=train)
         #调用train_writer的add_summary方法将训练过程以及训练步数保存
         writer.add_summary(train_summary,i)
         valid_summary = sess.run(validmerged,feed_dict=test)
         writer.add_summary(valid_summary,i)

         distance=sess.run((prediction-ys)*ytrainStd,feed_dict=train)
         accGeneral=acc(distance)
         acc_history.append(accGeneral)
         Testdistance=sess.run((prediction-ys)*ytrainStd,feed_dict=test)
         accGeneral=acc(Testdistance)
         acc_Testhistory.append(accGeneral)
#         acc_summary = scalar_summary('acc',accGeneral)
#         validacc_summary = scalar_summary('validacc',accGeneral)
#         trainAccmerged = merge_summary([acc_summary])
#         validAccmerged = merge_summary([validacc_summary])
#         writer.add_summary(validacc_summary,i)
#         writer.add_summary(acc_summary,i)
         if i % 100 == 0:
    #             m=(i)%100
    #             n=(i+1)%100
    #             if m>n:
    #                 m=n-1
#             print(train)
             distanceTrain=sess.run(((prediction-ys)*ytrainStd),feed_dict=train)
             print("distanceTrain:\n",distanceTrain)
             distanceTest=sess.run((prediction-ys)*ytrainStd,feed_dict=test)
             print("distanceTest:\n",distanceTest)
             predict=sess.run(prediction*ytrainStd+ytrainMean,feed_dict=trainx)
             print('predictiontrain \n',predict)#*ytrainStd+ytrainMean
             predictest=sess.run(prediction*ytrainStd+ytrainMean,feed_dict=testx)
             print('predictionTest:\n',predictest)
             savetrain.save(sess,"ckpt/model.ckpt",global_step=i)

#    predict=sess.run(prediction,feed_dict={xs:xdata_test})
    #如果归一化了，结果得反归一
#    predict=predict*ytrainStd+ytrainMean
#    np.savetxt('predict.csv', predict*ytrainStd+ytrainMean, delimiter = ',')
#    error=(predict-ydata_test)*ytrainStd
#    np.savetxt('error.csv', error, delimiter = ',')
#    saver = tf.train.Saver()
    #saver.restore(sess, "/tmp/model.ckpt")
    costTrain=sess.run(loss, feed_dict=train)
    print(costTrain,np.min(costTrain_history))
    plt.plot ( range ( len ( costTrain_history ) ) ,costTrain_history )
    plt.axis ( [ 0,steps,0,100] )
    plt.xlabel ( 'training epochs' )
    plt.ylabel ( 'cost' )
    plt.title ( 'costTrain history' )
    plt.show ( )
    costTest=sess.run(loss, feed_dict=test)
    print(costTest,np.min(costTest_history))
    plt.plot ( range ( len ( costTest_history ) ) ,costTest_history )
    plt.axis ( [ 0,steps,0,1000] )
    plt.xlabel ( 'training epochs' )
    plt.ylabel ( 'cost' )
    plt.title ( 'costTest history' )
    plt.show ( )
# with tf.Session() as sess:
#     saver.restore(sess,"../model.ckpt")

#     print(sess.run(prediction,feed_dict={xs:xdata_train})