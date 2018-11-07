#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 17:26:29 2018

@author: iloyu
"""
from neural import NeuralNet
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import *

#tf.reset_default_graph()
#merged_summary=tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,BPNet))
Numsize=490#样本数
input_dimension=165 #特征数
lrate=0.002 #学习速率
batch_size =0#100#训练batch的大小
normalized=1#是否归一化
steps=80000#迭代次数
output_dimension=165#属性
num_train=400
num_valid=89
num_test=90
channel=0 #x,y,z
total=400#训练样本数

hidden_layers1=input_dimension
hidden_layers2=1024
hidden_layers3=1024
hidden_layers4=output_dimension#1024#1024
hidden_layers5=2048
hidden_layers6=512#dimension
hidden_layers7=256
hidden_layers8=output_dimension
hidden=[hidden_layers2,hidden_layers3]

"""
输入（490，165）X，（490，165）Y
"""
inputX= pd.read_csv('XDATAFINAL.csv',sep=',')
inputY=pd.read_csv('YDATAFINAL.csv',sep=',')
x_data =inputX.values#[:,0:dimension]
y_data=inputY.values#[:,0:dimension]

mask=np.random.choice(Numsize-1,num_train,replace=False)
xdata_train =x_data[mask,:]
ydata_train =y_data[mask,:]
maskValid=np.random.choice(list(set(range(0,Numsize-1)).difference(set(mask))),
                          num_valid,replace=False)
xdata_test = x_data[maskValid, :]
ydata_test = y_data[maskValid, :]
xtrainMean=np.mean(xdata_train,axis=0)
xtrainStd=np.std(xdata_train,axis=0)
ytrainMean=np.mean(ydata_train,axis=0)
ytrainStd=np.std(ydata_train,axis=0)

"""
均值方差归一化，范围（-1，1）
"""
def normalization1(xdata_train,ydata_train,xdata_test,ydata_test):
    xdata_train-=xtrainMean
    xdata_train/=xtrainStd

    xdata_test-=xtrainMean
    xdata_test/=xtrainStd

    ydata_train-=ytrainMean
    ydata_train/=ytrainStd

    ydata_test-=ytrainMean
    ydata_test/=ytrainStd

normalization1(xdata_train,ydata_train,xdata_test,ydata_test)
"""
fine tune
"""
best_val=0
results={}
learning_rates=[1e-4]
regularization_strengths=[1e-3,5e-3]
for lr in learning_rates:
    for reg in regularization_strengths:
        net=NeuralNet(input_dimension,2,hidden,output_dimension)
        stats=net.train(xdata_train,ydata_train,xdata_test,ydata_test,num_iters=80000,batch_size=num_valid,learning_rate=lr,learning_rate_decay=0.95,reg=reg,verbose=False)
        predit=net.predict(xdata_test)*xtrainStd+xtrainMean

        val_acc=(np.abs((net.predict(xdata_test)-ydata_test)*xtrainStd)<1).sum(axis=1).mean()
        if val_acc>best_val:
            best_val=val_acc
            best_net=net
        results[(lr,reg)]=val_acc
for lr,reg in sorted(results):
    val_acc1=results[(lr,reg)]
    np.savetxt('finetune.csv', predit, delimiter = ',')
    print ('lr %f reg %f val accuracy %f ' %(lr,reg,val_acc1))
print ('best validation accuracy achieved during cross-validation:%f' %best_val)

plt.plot(stats['loss_history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training Loss history')
plt.show()
plt.subplot(2, 1, 1)
plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(stats['train_acc_history'], label='train')
plt.plot(stats['val_acc_history'], label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.show()
#test_acc = (net.predict(X_test_feats) == y_test).mean()
#print test_acc