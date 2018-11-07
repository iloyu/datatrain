#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 20:17:15 2018

@author: wanghao
"""
import tensorflow as tf
import numpy as np
from model import regression
from glob import glob
import os
from scipy import io
import time

#import random
tf.reset_default_graph()
#用try...except...避免因版本不同出现导入错误问题
try:
  #image_summary = tf.image_summary
  scalar_summary = tf.scalar_summary
  #histogram_summary = tf.histogram_summary
  merge_summary = tf.merge_summary
  SummaryWriter = tf.train.SummaryWriter
except:
  #image_summary = tf.summary.image
  scalar_summary = tf.summary.scalar
  #histogram_summary = tf.summary.histogram
  merge_summary = tf.summary.merge
  SummaryWriter = tf.summary.FileWriter



#预先给需要的变量提供占位符
inputs = tf.placeholder(tf.float32, [None,166])
labels = tf.placeholder(tf.float32, [None,165])
#inputs_realdata = tf.placeholder(tf.float32, [None,165])


#batchsize = tf.placeholder(tf.float32)
#absoluteError = tf.placeholder(tf.float32,[None,55])

#outputs = tf.placeholder(tf.float32, [None,150])

## 定义dropout的placeholder
#keep_prob = tf.placeholder(tf.float32)


#输出模型的预测结果
outputs = regression(inputs)

#dist_pre = outputs[:,165]
outputs_tmp = tf.reshape(outputs,[-1,55,3])
inputs_tmp = tf.reshape(inputs[:,0:165],[-1,55,3])
labels_tmp = tf.reshape(labels,[-1,55,3])

d_pre =  tf.sqrt(tf.reduce_sum(tf.square(outputs_tmp - inputs_tmp), reduction_indices = 2))
#d_pre = tf.reduce_mean(dist_pre)
#d_real = tf.sqrt(tf.reduce_sum(tf.square(labels_tmp - inputs_tmp), reduction_indices = 2))
d_real = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(labels_tmp - inputs_tmp), reduction_indices = 2)), reduction_indices = 1)
#d_real is 1-dim
d_real = tf.reshape(d_real,[-1,1])

#d_pre_1 = tf.zeros([200,45])
d_pre_1 = d_pre[:,0:6]
d_pre_2 = d_pre[:,7:13]
d_pre_3 = d_pre[:,17:30]
d_pre_4 = d_pre[:,31:37]
d_pre_5 = d_pre[:,41:55]
d_pre_update = tf.concat([d_pre_1,d_pre_2,d_pre_3,d_pre_4,d_pre_5],axis = 1)
#d_real is 2-dim
#tf.reduce_mean()函数的第二个参数为0，则表示第一维的元素取平均值，即每一列求平均值
#loss = tf.reduce_mean(tf.abs(d_pre - d_real)/d_real) + 0.1*tf.reduce_mean(tf.square(outputs - labels)) + 0.0001*tf.reduce_mean(tf.square(outputs - inputs_realdata))
#loss = tf.reduce_mean(tf.abs(d_pre - d_real))
#计算预测值和真实值之间的误差
lamb = 0.00001#0.0001
loss = tf.reduce_mean(tf.square(outputs - labels)) + lamb*tf.reduce_mean(tf.square(d_pre_update - d_real))#lamb*tf.reduce_mean(tf.square(outputs - inputs_realdata)) #+ gama*tf.reduce_mean(tf.square(absoluteError))
loss_summary = scalar_summary('loss',loss)
validloss_summary = scalar_summary('validloss',loss)

#t_vars = tf.trainable_variables()
#
#train_vars = [var for var in t_vars if 'w' in var.name]

#运行过程中，给定所需的显存空间，避免一次运行占用所有的显存空间
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#运用梯度下降算法，以0.1的学习速率最小化损失
train = tf.train.AdamOptimizer(0.00001).minimize(loss)

savetrain = tf.train.Saver(max_to_keep = 50000)
##weight_clipping
#clip_D = [p.assign(tf.clip_by_value(p,-0.001,0.001)) for p in train_vars]
#这种裁剪引用于WGAN里面，将权重的范围定在（-0.001,0.001），而我们的实验是想要将训练过程中的过小的权重值裁剪，恰好相反。

#train model
init = tf.global_variables_initializer()#初始化所有变量
sess = tf.Session(config = config)
sess.run(init)

trainmerged = merge_summary([loss_summary])
validmerged = merge_summary([validloss_summary])

writer = SummaryWriter('./logs',sess.graph)

#validation
batchsize_valid = 96
validDataPath = glob(os.path.join('./data/validbrain/*.mat'))
validLabelPath = glob(os.path.join('./data/validface/*.mat'))

validDataFiles = validDataPath[0:batchsize_valid]
validDataDict = [ io.loadmat(validDataFile) for validDataFile in validDataFiles]
validDataList = [ [] for i in range(batchsize_valid)]
for i in range(batchsize_valid):
    validDataList[i] = validDataDict[i]['brain']
validData1 = np.array(validDataList).astype(np.float32)[:,:,:]#trainData1 = 4*5*10*3
validData = np.reshape(validData1, [batchsize_valid,165], 'C')#trainData = 4*150;
#'C'按照通道的顺序进行reshape，第一个坐标的0、1、 2通道以此类推

validLabelFiles = validLabelPath[0:batchsize_valid]
validLabelDict = [ io.loadmat(validLabelFile) for validLabelFile in validLabelFiles]
validLabelList = [ [] for i in range(batchsize_valid)]
for i in range(batchsize_valid):
    validLabelList[i] = validLabelDict[i]['face']
validLabel1 = np.array(validLabelList).astype(np.float32)[:,:,:]#trainLabel1 = 4*5*10*3
validLabel = np.reshape(validLabel1, [batchsize_valid,165], 'C')#trainLabel = 4*150
#validLabelres = validLabel - validData

#计算预测数的坐标对应特征点的厚度并与真是的厚度进行比较
validData_temp = np.zeros([batchsize_valid, 166])
mean_dist_valid = 0.0


for i in range(batchsize_valid):
    for j in range(55):
        dist_valid = np.sqrt(np.square(validLabel1[i,j,1]-validData1[i,j,1]) + np.square(validLabel1[i,j,2]-validData1[i,j,2]) + np.square(validLabel1[i,j,0]-validData1[i,j,0]))
        mean_dist_valid = mean_dist_valid + dist_valid
    mean_dist_valid = mean_dist_valid/55.0
    validData_temp[i,0:165] = validData[i,0:165]#从下标0开始取165个数据
    validData_temp[i,165] = mean_dist_valid



##recover model
##batchsize = 4
#checkpoint_dir = './checkpoint'
#print(" [*] Reading checkpoints...")
#model_dir = "{}_{}_{}_{}".format('trainbrain',200,165,1)
#checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
#
#ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
#if ckpt and ckpt.model_checkpoint_path:
#    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
#    saver = tf.train.Saver()
#    saver.restore(sess, './checkpoint/trainbrain_200_165_1/regression.model-1440000')
#    print(" [*] Success to load!")
#else:
#    print(" [*] Failed to find a checkpoint")


counter = 1#50000+1
start_time = time.time()

##将打乱的顺序预先保存在trianindex列表中，达到固定打乱顺序的功能
#indexList = [ [] for i in range(20)]
#for i in range(20):
#    index = range(55)
#    random.shuffle(index)
#    indexList[i] = index

#迭代训练1000次
for epoch in range(100000):
    batchsize = 10
    #获取只是数据和标签数据的路径
    trainDataPath = glob(os.path.join('./data/trainbrain/*.mat'))
    trainLabelPath = glob(os.path.join('./data/trainface/*.mat'))
    for idx in range(round(len(trainDataPath)/batchsize)):#100 = len(trainDataPAth/batchsize)

        trainDataFiles = trainDataPath[idx*batchsize:(idx+1)*batchsize]
        trainDataDict = [ io.loadmat(trainDataFile) for trainDataFile in trainDataFiles]
        trainDataList = [ [] for i in range(batchsize)]
        for i in range(batchsize):
            trainDataList[i] = trainDataDict[i]['brain']
        trainData1 = np.array(trainDataList).astype(np.float32)[:,:,:]#trainData1 = 4*5*10*3

        trainData = np.reshape(trainData1, [batchsize,165], 'C')#trainData = 4*150;
        #'C'按照通道的顺序进行reshape，第一个坐标的0、1、 2通道以此类推

        trainLabelFiles = trainLabelPath[idx*batchsize:(idx+1)*batchsize]
        trainLabelDict = [ io.loadmat(trainLabelFile) for trainLabelFile in trainLabelFiles]
        trainLabelList = [ [] for i in range(batchsize)]
        for i in range(batchsize):
            trainLabelList[i] = trainLabelDict[i]['face']
        trainLabel1 = np.array(trainLabelList).astype(np.float32)[:,:,:]#trainLabel1 = 4*5*10*3

        trainLabel = np.reshape(trainLabel1, [batchsize,165], 'C')#trainLabel = 4*150
#        trainLabelres = trainLabel - trainData
        #reshape后，要保证trainData和trainLabel中的坐标数据一一对应

        #计算预测数的坐标对应特征点的厚度并与真是的厚度进行比较
        trainData_temp = np.zeros([batchsize, 166])
        mean_dist_real = 0.0
        for i in range(batchsize):
            for j in range(55):
                dist_real = np.sqrt(np.square(trainLabel1[i,j,1]-trainData1[i,j,1]) + np.square(trainLabel1[i,j,2]-trainData1[i,j,2]) + np.square(trainLabel1[i,j,0]-trainData1[i,j,0]))
                mean_dist_real = mean_dist_real + dist_real
            mean_dist_real = mean_dist_real/55
            trainData_temp[i,0:165] = trainData[i,0:165]
            trainData_temp[i,165] = mean_dist_real

        #迭代更新并且计算损失
        summary, _,trainLoss, outres = sess.run([trainmerged,train, loss, outputs], feed_dict={inputs:trainData_temp, labels:trainLabel})
        writer.add_summary(summary,counter)

        summary_valid, validLoss = sess.run([validmerged, loss], feed_dict={inputs:validData_temp, labels:validLabel})
        writer.add_summary(summary_valid,counter) #clip_D,
        print(validLoss)

        counter += 1
        print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, loss: %.8f," %(epoch,10, idx, 1, time.time() - start_time, trainLoss))

        #保存训练过程中的参数
        if np.mod(counter, 200) == 0:
            #save('./checkpoint', counter)
            model_name = "regression.model"
            model_dir = "{}_{}_{}_{}".format('trainbrain',batchsize,165,1)
            checkpoint_dir = os.path.join('./checkpoint', model_dir)

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            #savetrain = tf.train.Saver()
            savetrain.save(sess,
                os.path.join(checkpoint_dir, model_name),
                global_step=counter)
            #每50次进行一次裁剪
            #saver = tf.train.Saver()
            checkpoint_path = os.path.join(checkpoint_dir, model_name+'-' + str(counter))
            savetrain.restore(sess, checkpoint_path)


            all_vars = tf.trainable_variables()
            train_vars = [var for var in all_vars if 'w' in var.name]
            crop = [var.assign(tf.where(tf.abs(var)<0.0001, 0*var, var)) for var in train_vars]
            sess.run(crop)
            savetrain.save(sess, checkpoint_path )


#        if np.mod(counter,10000) == 0:
#            model_name = "regression.model"
#            model_dir = "{}_{}_{}_{}".format('trainbrain',batchsize,165,1)
#            checkpoint_dir = os.path.join('./checkpoint', model_dir)
#            #每50次进行一次裁剪
#            saver = tf.train.Saver()
#            checkpoint_path = os.path.join(checkpoint_dir, model_name+'-' + str(counter))
#            saver.restore(sess, checkpoint_path)
#
#
#            all_vars = tf.trainable_variables()
#            train_vars = [var for var in all_vars if 'w' in var.name]
#            crop = [var.assign(tf.where(tf.abs(var)<0.0001, 0*var, var)) for var in train_vars]
#            sess.run(crop)
#            saver.save(sess, checkpoint_path )







