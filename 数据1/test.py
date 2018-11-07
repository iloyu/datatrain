#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 18:21:22 2018

@author: wanghao
"""

import tensorflow as tf
from model import regression
import numpy as np
from glob import glob
import os
from scipy import io 
import time
import re

batchsize = 100


## 定义dropout的placeholder
#keep_prob = tf.placeholder(tf.float32)

#预先给需要的变量提供占位符
inputs_test = tf.placeholder(tf.float32, [None,166])
inputs_testrealdata = tf.placeholder(tf.float32, [None,165])
label_test = tf.placeholder(tf.float32, [None,165])
#输出模型的预测结果
outputs_test = regression(inputs_test)

outputs_tmp = tf.reshape(outputs_test,[-1,55,3])
inputs_tmp = tf.reshape(inputs_test[:,0:165],[-1,55,3])
labels_tmp = tf.reshape(label_test,[-1,55,3])

d_pre = tf.sqrt(tf.reduce_sum(tf.square(outputs_tmp - inputs_tmp), reduction_indices = 2))

d_real = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(labels_tmp - inputs_tmp), reduction_indices = 2)), reduction_indices = 1)
d_real = tf.reshape(d_real,[-1,1])
#loss = tf.reduce_mean(tf.abs(d_pre - d_real)/d_real)

#计算预测值和真实值之间的误差
lamb = 0.00001#0.0001
loss = tf.reduce_mean(tf.square(outputs_test - label_test )) + lamb*tf.reduce_mean(tf.square(d_pre - d_real))#lamb*tf.reduce_mean(tf.square(outputs_test - inputs_testrealdata)) 
##运用梯度下降算法，以0.1的学习速率最小化损失
#train = tf.train.AdamOptimizer(0.1).minimize(loss)

#test model
#init = tf.global_variables_initializer()#初始化所有变量
sess = tf.Session()
#sess.run()

#recover model
checkpoint_dir = './checkpoint'
print(" [*] Reading checkpoints...")
model_dir = "{}_{}_{}_{}".format('trainbrain',122,165,1)
checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt and ckpt.model_checkpoint_path:
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    saver = tf.train.Saver()
    saver.restore(sess, './checkpoint/trainbrain_122_165_1/regression.model-30000')
    print(" [*] Success to load!")
else:
    print(" [*] Failed to find a checkpoint")
#    
start_time = time.time()    
#获取test数据和标签数据的路径 
testDataPath = glob(os.path.join('./data/testbrain/*.mat'))
testLabelPath = glob(os.path.join('./data/testface/*.mat'))
for idx in range(len(testDataPath)/batchsize):#50 = len(testDataPAth/batchsize)
    testDataFiles = testDataPath[idx*batchsize:(idx+1)*batchsize]
    testDataDict = [ io.loadmat(testDataFile) for testDataFile in testDataFiles]
    testDataList = [ [] for i in range(batchsize)]
    for i in range(batchsize):
        testDataList[i] = testDataDict[i]['brain']
    testData1 = np.array(testDataList).astype(np.float32)[:,:,:]#trainData1 = 4*5*10*3
    testData = np.reshape(testData1, [batchsize,165], 'C')#trainData = 4*150; 
    
    testLabelFiles = testLabelPath[idx*batchsize:(idx+1)*batchsize]
    testLabelDict = [ io.loadmat(testLabelFile) for testLabelFile in testLabelFiles]
    testLabelList = [ [] for i in range(batchsize)]
    for i in range(batchsize):
        testLabelList[i] = testLabelDict[i]['face']
    testLabel1 = np.array(testLabelList).astype(np.float32)[:,:,:]#trainLabel1 = 4*5*10*3
    testLabel = np.reshape(testLabel1, [batchsize,165], 'C')#trainLabel = 4*150
#    testLabelres = testLabel - testData
    
    #计算预测数的坐标对应特征点的厚度并与真是的厚度进行比较
    testData_temp = np.zeros([batchsize, 166])
    mean_dist_test = 0.0
    
    
    for i in range(batchsize):
        for j in range(55):
            dist_test = np.sqrt(np.square(testLabel1[i,j,1]-testData1[i,j,1]) + np.square(testLabel1[i,j,2]-testData1[i,j,2]) + np.square(testLabel1[i,j,0]-testData1[i,j,0]))
            mean_dist_test = mean_dist_test + dist_test
        mean_dist_test = mean_dist_test/55.0
        testData_temp[i,0:165] = testData[i,0:165]#从下标0开始取165个数据
        testData_temp[i,165] = mean_dist_test
        
#    #计算预测数的坐标对应特征点的厚度并与真是的厚度进行比较
#    testData_temp = np.zeros([batchsize, 166])
#    mean_dist_test = 13
#    for i in range(batchsize):
##        for j in range(55):
##            dist_test = np.sqrt(np.square(testLabel1[i,j,1]-testData1[i,j,1]) + np.square(testLabel1[i,j,2]-testData1[i,j,2]) + np.square(testLabel1[i,j,0]-testData1[i,j,0]))
##            mean_dist_test = mean_dist_test + dist_test
##        mean_dist_test = mean_dist_test/55
#        testData_temp[i,0:165] = testData[i,0:165]
#        testData_temp[i,165] = mean_dist_test
    
    
    out_test, test_loss = sess.run([outputs_test, loss], feed_dict={inputs_test:testData_temp, label_test:testLabel, inputs_testrealdata:testData})
    print("time: %4.4f, loss: %.8f,", time.time() - start_time, test_loss)
    
#    pretest[:,0:165] = out_test_res[:,0:165] + testData
#    pretest[:,165] = out_test_res[:,165]
#    prelabel = pretest
    prelabel = out_test
    #prelabel = out_test_res
    print(testDataFiles)
    prelabel = np.reshape(prelabel,[batchsize,55,3])
    for i in range(batchsize):
        savepath = './result/'
        array = prelabel[i,:,:]
        path = str(testDataFiles[i])
        str1 = path[17]
        for j in range(18,len(path)):
            str1 = str1 + path[j]
        savepath = savepath + str1
        io.savemat(savepath, {'face':array})
     
    #将98个测试样本的55个特征的厚度以及对应特征的厚度的均值，方差，绝对误差，相对误差保存在（98+4）*55的二维矩阵中。 
    distArray_pre =  np.zeros([batchsize+4, 55])
    distArray_real =  np.zeros([batchsize+4, 55])
    distArray_res = np.zeros([batchsize+4, 55])
    
    absoluteError = np.zeros([batchsize+2, 55])
    relativeError = np.zeros([batchsize+2, 55])
    distreal = 0.0
    distpre = 0.0
    for batch in range(batchsize):
        for col in range(55):
             distreal = np.sqrt(np.square(testLabel1[batch,col,0]-testData1[batch,col,0]) + np.square(testLabel1[batch,col,1]-testData1[batch,col,1]) + np.square(testLabel1[batch,col,2]-testData1[batch,col,2]))
             distpre = np.sqrt(np.square(prelabel[batch,col,0]-testData1[batch,col,0]) + np.square(prelabel[batch,col,1]-testData1[batch,col,1]) + np.square(prelabel[batch,col,2]-testData1[batch,col,2]))
             distArray_real[batch,col] = distreal
             distArray_pre[batch,col] = distpre 
             
             absoluteError[batch,col] = np.abs(distreal - distpre)
             relativeError[batch,col] = np.abs(distreal - distpre)/distreal#*100
    
    distArray_res = distArray_real - distArray_pre
    
    for col in range(55):
        distArray_pre[batchsize,col]= np.mean(distArray_pre[0:batchsize,col]) 
        distArray_real[batchsize,col]= np.mean(distArray_real[0:batchsize,col]) 
        distArray_pre[batchsize+1,col]= np.var(distArray_pre[0:batchsize,col]) 
        distArray_real[batchsize+1,col]= np.var(distArray_real[0:batchsize,col]) 
        
        distArray_res[batchsize,col] = np.mean(distArray_res[0:batchsize,col])
        distArray_res[batchsize+1,col] = np.var(distArray_res[0:batchsize,col])
        
        
        absoluteError[batchsize,col] = np.mean(absoluteError[0:batchsize,col])   
        relativeError[batchsize,col] = np.mean(relativeError[0:batchsize,col]) 
        absoluteError[batchsize+1,col] = np.var(absoluteError[0:batchsize,col])   
        relativeError[batchsize+1,col] = np.var(relativeError[0:batchsize,col]) 

        distArray_pre[batchsize+2,col]= np.mean(absoluteError[0:batchsize,col]) 
        distArray_pre[batchsize+3,col]= np.mean(relativeError[0:batchsize,col]) 
        

    print("error:")
    for row in range(98):
        for col in range(1):
            print(relativeError[row,col])
        
    print("absolute error:")
    num = 0
    for col in range(55):
        if distArray_pre[100,col]<2.0 and distArray_pre[101,col]<0.1:
            num = num + 1
            print("%d",col,distArray_pre[100,col],distArray_pre[101,col])    
    print("relative error:")
    number = 0
    for col in range(55):
        if distArray_pre[101,col]<10.0:
            number = number + 1
        print("%d",i,distArray_pre[101,col])
        
    name1 = 'distArrayPre'
    path1 = './errorAnalysis/' + name1
    io.savemat(path1, {'distPre':distArray_pre})
    
    name2 = 'distArrayReal'
    path2 = './errorAnalysis/' + name2
    io.savemat(path2, {'distReal':distArray_real})
    
    name3 = 'absoluteError'
    path3 = './errorAnalysis/' + name3
    io.savemat(path3, {'absolute':absoluteError})
    
    name4 = 'relativeError'
    path4 = './errorAnalysis/' + name4
    io.savemat(path4, {'relative':relativeError})
    
    name5 = 'distArrayRes'
    path5 = './errorAnalysis/' + name5
    io.savemat(path5, {'distres':distArray_res})
        
    
    #print real and predict distance, absolute error distance  and mean absolute error distance of per character
    per_distreal = 0.0
    per_distpre = 0.0
    per_meandistreal = 0.0
    per_meandistpre = 0.0
    
    mean_absoluteError = 0.0
    absoluteError = np.zeros([batchsize, 55,1])
    
    mean_relativeError = 0.0
    relativeError = np.zeros([batchsize, 55,1])
    for batch in range(batchsize):
        for row in range(55):
             per_distreal = np.sqrt(np.square(testLabel1[batch,row,0]-testData1[batch,row,0]) + np.square(testLabel1[batch,row,1]-testData1[batch,row,1]) + np.square(testLabel1[batch,row,2]-testData1[batch,row,2]))
             print('batch:%d, real_character:%d, per_distreal', batch, row, per_distreal)
             per_distpre = np.sqrt(np.square(prelabel[batch,row,0]-testData1[batch,row,0]) + np.square(prelabel[batch,row,1]-testData1[batch,row,1]) + np.square(prelabel[batch,row,2]-testData1[batch,row,2]))
             print('batch:%d, pre_character:%d, per_distpre', batch, row, per_distpre)
             
             #compute absolute error and then store in the array: absoluteError
             absoluteError[batch,row,0] = np.abs(per_distreal - per_distpre)
             mean_absoluteError = mean_absoluteError + np.abs(per_distreal - per_distpre)
             
             #compute relative error and then stroe in the array:relativeError
             relativeError[batch,row,0] = (np.abs(per_distreal - per_distpre)/per_distreal)*100
             mean_relativeError = mean_relativeError + relativeError[batch,row,0]
             
             print('per_distreal - per_distpre: %4.4f', np.abs(per_distreal - per_distpre))
             
             per_meandistreal = per_meandistreal + per_distreal
             per_meandistpre = per_meandistpre + per_distpre
        per_meandistreal = per_meandistreal/55 
        print('batch:%d, real_meandist:%d', batch, per_meandistreal)
        per_meandistpre = per_meandistpre/55 
        print('batch:%d, pre_meandist:%d', batch, per_meandistpre)
        
        mean_absoluteError = mean_absoluteError/55
        print('mean_absoluteError:4.4f', mean_absoluteError)
        mean_relatibe = mean_absoluteError/55
        print('mean_relativeError:%4.4f', mean_relativeError)
        
        #store
        errorpath1 = './absoluteError/' + str(batch)
        io.savemat(errorpath1, {'error':absoluteError[batch,:,:]})
        
        errorpath2 = './relativeError/' + str(batch)
        io.savemat(errorpath2, {'error':relativeError[batch,:,:]})
#    print(prelabel[0,15,:])
#    print(testLabel1[0,15,:])
#    print(testData1[0,15,:])
#    
#    
#    np.sqrt(np.square(testLabel1[batch,row,0]-testData1[batch,row,0]) + np.square(testLabel1[batch,row,1]-testData1[batch,row,1]) + np.square(testLabel1[batch,row,2]-testData1[batch,row,2]))
#    
#    

#统计44样本对应的2420个特征点对应的相对误差分布的直方图
relativeErrorHist = np.zeros([11])
for batch in range(batchsize):
    for row in range(55):
        if relativeError[batch,row,0] <10.0:
            relativeErrorHist[0] = relativeErrorHist[0] + 1
        elif relativeError[batch,row,0] <20.0:
            relativeErrorHist[1] = relativeErrorHist[1] + 1
        elif relativeError[batch,row,0] <30.0:
            relativeErrorHist[2] = relativeErrorHist[2] + 1
        elif relativeError[batch,row,0] <40.0:
            relativeErrorHist[3] = relativeErrorHist[3] + 1
        elif relativeError[batch,row,0] <50.0:
            relativeErrorHist[4] = relativeErrorHist[4] + 1
        elif relativeError[batch,row,0] <60.0:
            relativeErrorHist[5] = relativeErrorHist[5] + 1
        elif relativeError[batch,row,0] <70.0:
            relativeErrorHist[6] = relativeErrorHist[6] + 1
        elif relativeError[batch,row,0] <80.0:
            relativeErrorHist[7] = relativeErrorHist[7] + 1
        elif relativeError[batch,row,0] <90.0:
            relativeErrorHist[8] = relativeErrorHist[8] + 1
        elif relativeError[batch,row,0] <100.0:
            relativeErrorHist[9] = relativeErrorHist[9] + 1
        else:
            relativeErrorHist[10] = relativeErrorHist[10] + 1
io.savemat('./hist.mat', {'hist':relativeErrorHist})        
            
            
#relativeErrorHist = np.zeros([1,11])
#for batch in range(batchsize):
#    for row in range(55):
#        if relativeError[batch,row,0] <10.0:
#            relativeErrorHist[0,0] = relativeErrorHist[0,0] + 1
#        elif relativeError[batch,row,0] <20.0:
#            relativeErrorHist[0,1] = relativeErrorHist[0,1] + 1
#        elif relativeError[batch,row,0] <30.0:
#            relativeErrorHist[0,2] = relativeErrorHist[0,2] + 1
#        elif relativeError[batch,row,0] <40.0:
#            relativeErrorHist[0,3] = relativeErrorHist[0,3] + 1
#        elif relativeError[batch,row,0] <50.0:
#            relativeErrorHist[0,4] = relativeErrorHist[0,4] + 1
#        elif relativeError[batch,row,0] <60.0:
#            relativeErrorHist[0,5] = relativeErrorHist[0,5] + 1
#        elif relativeError[batch,row,0] <70.0:
#            relativeErrorHist[0,6] = relativeErrorHist[0,6] + 1
#        elif relativeError[batch,row,0] <80.0:
#            relativeErrorHist[0,7] = relativeErrorHist[0,7] + 1
#        elif relativeError[batch,row,0] <90.0:
#            relativeErrorHist[0,8] = relativeErrorHist[0,8] + 1
#        elif relativeError[batch,row,0] <100.0:
#            relativeErrorHist[0,9] = relativeErrorHist[0,9] + 1
#        else:
#            relativeErrorHist[0,10] = relativeErrorHist[0,10] + 1


for i in range(11):
    print(relativeErrorHist[i])
    
fig1 = plt.figure(2)
rects =plt.bar(left = (0.2,1),height = (15.,0.5),width = 0.2)
plt.title('Pe')
plt.show()    


import matplotlib.pyplot as plt
plt.hist(relativeErrorHist)
#for batch in range(batchsize):
#    print(batch,relativeError[batch,:,:].max())
#    for i in range(55):
#        print(relativeError[batch,:,0])


    
    
    
    
    
    
    
    