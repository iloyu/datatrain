#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 14:26:29 2018

@author: iloyu
"""


import os
import tensorflow as tf
import numpy as np
import pandas as pd
from model import regression
import matplotlib.pyplot as plt
from numpy import *

tf.reset_default_graph()
Numsize=490#样本数
feature=165 #特征数
#lrate=1e-6 #学习速率1e-4
batch_size =0#100#训练batch的大小
normalized=1#是否归一化
steps=100000#迭代次数

channel=3 #x,y,z
total=400#训练样本数
num_train=400
num_valid=89
p=1
regular=10.0

train_start = 0
train_end = int(np.floor(0.8*Numsize))
test_start = train_end
test_end = Numsize-1


cols=[4,6,7,9,12,13,18,28,30,31,33,36,37,38]+[59, 61, 62, 64, 67, 68, 73, 83, 85, 86, 88, 91, 92, 93]+[114, 116, 117, 119, 122, 123, 128, 138, 140, 141, 143, 146, 147, 148]
dimension=(int)(len(cols)/3)#属性
hidden_layers1=dimension
hidden_layers2=1024
hidden_layers3=1024
hidden_layers4=2048#dimension#1024#1024
hidden_layers5=2048
hidden_layers6=512
hidden_layers7=256
hidden_layers8=dimension
inputX= pd.read_csv('XDATAFINAL.csv',sep=',',usecols=cols)
inputY=pd.read_csv('YDATAFINAL.csv',sep=',',usecols=cols)
#inputX= pd.read_csv('xDataR2.txt',header=None,sep=',')
#inputY=pd.read_csv('yDataR2.txt',header=None,sep=',')
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


colSize=inputX.columns.size
xf=[i for i in range(colSize) if i%3==0]
xdata_train1=xdata_train[:,xf]
ydata_train1=ydata_train[:,xf]
xdata_test1=xdata_test[:,xf]
ydata_test1=ydata_test[:,xf]

#y channel
xf=[i for i in range(colSize) if i%3==1]
xdata_train2=xdata_train[:,xf]
ydata_train2=ydata_train[:,xf]
xdata_test2=xdata_test[:,xf]
ydata_test2=ydata_test[:,xf]
#z channel
xf=[i for i in range(colSize) if i%3==2]
xdata_train3=xdata_train[:,xf]
ydata_train3=ydata_train[:,xf]
xdata_test3=xdata_test[:,xf]
ydata_test3=ydata_test[:,xf]

xdata_trainMix=np.concatenate((xdata_train1,xdata_train2,xdata_train3),1)
xdata_testMix=np.concatenate((xdata_test1,xdata_test2,xdata_test3),1)
ydata_trainMix=np.concatenate((ydata_train1,ydata_train2,ydata_train3),1)
ydata_testMix=np.concatenate((ydata_test1,ydata_test2,ydata_test3),1)


xs1 = tf.placeholder(tf.float32, [None,dimension],name="x_traindata1")
xs2 = tf.placeholder(tf.float32, [None,dimension],name="x_traindata2")
xs3 = tf.placeholder(tf.float32, [None,dimension],name="x_traindata3")
ys= tf.placeholder(tf.float32, [None,dimension*3],name="y_traindata")

def LeakyRelu(x, leak=0.2, name="LeakyRelu"):
 with tf.variable_scope(name):
     f1 = 0.5 * (1 + leak)
     f2 = 0.5 * (1 - leak)
     return f1 * x + f2 * tf.abs(x)
#prediction = regression(xs)

def add_layer(wname,bname,inputs, in_size, out_size, activation_function=None):
    with tf.variable_scope("BPNet") :
        weights = tf.get_variable(wname,shape=[in_size,out_size], initializer=tf.contrib.layers.xavier_initializer(), regularizer = tf.contrib.layers.l2_regularizer(regular))#xavier_initializer()variance_scaling_initializer()
        b = tf.get_variable(bname, shape=[1,out_size], initializer=tf.zeros_initializer())
        Wx_plus_b = tf.matmul(inputs, weights) + b
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
       # tf.histogram_summary(wname + '/outputs', outputs)#变量跟踪
        return weights,outputs

## 构建输入层到隐藏层,假设隐藏层有 hidden_layers 个神经元

w11,h11=add_layer('w11','b11',xs1,dimension,hidden_layers1)
#h11 = tf.nn.dropout(h11,p)
w21,h21=add_layer('w21','b21',h11,hidden_layers1,hidden_layers2,activation_function=tf.nn.relu)
#h21=tf.nn.dropout(h21,p)
w31,h31=add_layer('w31','b31',h21,hidden_layers2,hidden_layers3,activation_function=tf.nn.relu)
#h31=tf.nn.dropout(h31,p)
w41,h41=add_layer('w41','b41',h31,hidden_layers3,hidden_layers4,activation_function=tf.nn.relu)
#h41=tf.nn.dropout(h41,p)
w51,h51=add_layer('w51','b51',h41,hidden_layers4,hidden_layers5,activation_function=tf.nn.relu)
w61,prediction1=add_layer('w61','b61',h51,hidden_layers5,dimension)

w12,h12=add_layer('w12','b12',xs2,dimension,hidden_layers1)
#h12= tf.nn.dropout(h12,p)
w22,h22=add_layer('w22','b22',h12,hidden_layers1,hidden_layers2,activation_function=tf.nn.relu)
#h22 = tf.nn.dropout(h22,p)
w32,h32=add_layer('w32','b32',h22,hidden_layers2,hidden_layers3,activation_function=tf.nn.relu)
#h32= tf.nn.dropout(h32,p)
w42,h42=add_layer('w42','b42',h32,hidden_layers3,hidden_layers4,activation_function=tf.nn.relu)
#h42 = tf.nn.dropout(h42,p)
w52,h52=add_layer('w52','b52',h42,hidden_layers4,hidden_layers5,activation_function=tf.nn.relu)

w62,prediction2=add_layer('w62','b62',h52,hidden_layers5,dimension)

w13,h13=add_layer('w13','b13',xs3,dimension,hidden_layers1)
#h13 = tf.nn.dropout(h13,p)
w23,h23=add_layer('w23','b23',h13,hidden_layers1,hidden_layers2,activation_function=tf.nn.relu)
#h23 = tf.nn.dropout(h23,p)
w33,h33=add_layer('w33','b33',h23,hidden_layers2,hidden_layers3,activation_function=tf.nn.relu)
#h33 = tf.nn.dropout(h33,p)
w43,h43=add_layer('w43','b43',h33,hidden_layers3,hidden_layers4,activation_function=tf.nn.relu)
#h43 = tf.nn.dropout(h43,p)
w53,h53=add_layer('w53','b53',h43,hidden_layers4,hidden_layers5,activation_function=tf.nn.relu)
w63,prediction3=add_layer('w63','b63',h53,hidden_layers5,dimension)

prediction=tf.concat([prediction1,prediction2,prediction3],1)
ys1,ys2,ys3=tf.split(ys,3,1)

thickpre=tf.sqrt(tf.square(prediction1-xs1)+tf.square(prediction2-xs2)+tf.square(prediction3-xs3))
thickreal=tf.sqrt(tf.square(ys1-xs1)+tf.square(ys2-xs2)+tf.square(ys3-xs3))

loss=tf.reduce_sum(tf.abs(thickpre-thickreal)/tf.abs(thickreal))

scalar_summary = tf.summary.scalar
#histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter
loss_summary = scalar_summary('losscritical',loss)
validloss_summary =scalar_summary('validloss',loss)
#acc_summary =scalar_summary('acc',accurate)
#validacc_summary =scalar_summary('validacc',accurate)
trainmerged =merge_summary([loss_summary])
validmerged =merge_summary([validloss_summary])
savetrain = tf.train.Saver(max_to_keep = 1)
test={xs1:xdata_test1,xs2:xdata_test2,xs3:xdata_test3,ys:ydata_testMix}
testx={xs1:xdata_test1,xs2:xdata_test2,xs3:xdata_test3}
trainx={xs1:xdata_train1,xs2:xdata_train2,xs3:xdata_train3}
train={xs1:xdata_train1,xs2:xdata_train2,xs3:xdata_train3,ys:ydata_trainMix}
#imported_meta=tf.train.import_meta_graph("*.meta")
current_epoch = tf.Variable(0,trainable=False)
learning_rate = tf.train.exponential_decay(0.00001,
                                           current_epoch,
                                           decay_steps=steps,
                                           decay_rate=0.95,staircase=True)


#loss=tf.reduce_sum(tf.reduce_mean(tf.abs(thickpre-thickreal),0))
train_step =tf.train.AdamOptimizer(learning_rate).minimize(loss)
col1=int(inputX.columns.size/3)
col2=int(inputX.columns.size/3*2)
saver = tf.train.Saver(tf.global_variables())
moudke_file=tf.train.latest_checkpoint('./ckpt1/')

with tf.Session() as sess:
  ckpt='./ckpt1/'+"model.ckpt-"+str(80000)
#  saver.restore(sess,moudke_file)
  savetrain.restore(sess,moudke_file)
  sess.run(tf.global_variables_initializer())

#  imported_meta.restore(sess,tf.train.latest_checkpoint('./'))
  writer = SummaryWriter('./log1',sess.graph)

  MinMean=1000000000
  thickMatrixReal=np.ones(shape=(num_valid,0))
  SquareReal=np.square(ydata_testMix-xdata_testMix)
  for j in range(col1):
      thickMatrixReal=np.column_stack((thickMatrixReal,np.sqrt(np.add(np.add(SquareReal[:,j],SquareReal[:,j+col1]),SquareReal[:,j+col2]))))
      np.savetxt('RealthicknessC.csv', thickMatrixReal, delimiter = ',')

  for i in range(1,steps+1):
       current_epoch=i
       sess.run(train_step, feed_dict=train)
       costTrain=sess.run(loss, feed_dict=train)
#         costTrain_history.append(costTrain)
        #调用sess.run运行图，生成一步的训练过程数据
       train_summary = sess.run(trainmerged,feed_dict=train)
        #调用train_writer的add_summary方法将训练过程以及训练步数保存
       writer.add_summary(train_summary,i)

       costTest=sess.run(loss, feed_dict=test)
#         costTest_history.append(costTest)

       valid_summary = sess.run(validmerged,feed_dict=test)

       writer.add_summary(valid_summary,i)

       if i % 100 == 0:

           thickMatrix=np.ones(shape=(num_valid,0))

           predict=sess.run(prediction,feed_dict=trainx)

           predictest=sess.run(prediction,feed_dict=testx)
           Testdif=sess.run((prediction-ys),feed_dict=test)
           print("difTest:\n",Testdif)
           Square=np.square(predictest-xdata_testMix)
           for j in range(col1):
               thickMatrix=np.column_stack((thickMatrix,np.sqrt(np.add(np.add(Square[:,j],Square[:,j+col1]),Square[:,j+col2]))))


           dismean=np.mean(abs(thickMatrixReal-thickMatrix),0)
#             disStdTrain=np.std(abs(distance),0)
#             dismean=np.mean(abs(Testdistance),0)
           disStd=np.std(abs(thickMatrixReal-thickMatrix),0)
           disMeantmp=np.mean(dismean)
           if disMeantmp<MinMean:
             MinMean=disMeantmp
             Savedismean=dismean
             Savestd=disStd
             Savepre=predictest
             SaveMatrixpre=thickMatrix


           print("distanceTestMean&&Std:\n",dismean,disStd,"\nTestMean",disMeantmp)
           savetrain.save(sess,os.path.join("ckpt1/model80000.ckpt"),global_step=i)
           if i%2000 ==0:
             np.savetxt('dismeanC.csv', Savedismean, delimiter = ',')
             np.savetxt('disstdC.csv', Savestd, delimiter = ',')
             np.savetxt('originalC.csv', Savepre, delimiter = ',')
             np.savetxt('thicknessC.csv', SaveMatrixpre, delimiter = ',')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family']='sans-serif'
plt.rcParams['axes.unicode_minus'] = False


tmp=np.mean(thickMatrixReal,0)
plt.plot(tmp,Savedismean,'ro')

plt.xlabel ( u'55 mean 厚度 ' )
plt.ylabel ( u'55 thickness mean error in prediction ' )
plt.title ( u'relation between thickness and prediction error' )