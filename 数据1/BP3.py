#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 13:58:31 2018

@author: iloyu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 09:49:59 2018

@author: iloyu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 17:54:53 2018

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
#merged_summary=tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,BPNet))
Numsize=500#样本数
feature=165 #特征数
lrate=0.0001 #学习速率
batch_size =0#100#训练batch的大小
normalized=1#是否归一化
steps=1000000#迭代次数
dimension=55#属性
channel=3 #x,y,z
total=400#训练样本数


train_start = 0
train_end = int(np.floor(0.8*Numsize))
test_start = train_end
test_end = Numsize

hidden_layers1=dimension
hidden_layers2=512
hidden_layers3=1024
hidden_layers4=2048#dimension#1024#1024
hidden_layers5=2048
hidden_layers6=512
hidden_layers7=256
hidden_layers8=dimension


inputX= pd.read_csv('xDataR2.txt',header=None,sep=',')
inputY=pd.read_csv('yDataR2.txt',header=None,sep=',')
x_data =inputX.values#[:,0:dimension]
y_data=inputY.values#[:,0:dimension]

xdata_train =x_data[np.arange(train_start, train_end),:]
xdata_test = x_data[np.arange(test_start, test_end), :]
ydata_train =y_data[np.arange(train_start,train_end), :]
ydata_test = y_data[np.arange(test_start,test_end), :]
xtrainMean=np.mean(xdata_train,axis=0)
xtrainStd=np.std(xdata_train,axis=0)
ytrainMean=np.mean(ydata_train,axis=0)
ytrainStd=np.std(ydata_train,axis=0)
# print(xdata_test,xdata_test.shape)
# print(ydata_test,ydata_test.shape)
#均值方差归一化，范围（-1，1）
def normalization1(xdata_train,ydata_train,xdata_test,ydata_test):
    xdata_train-=xtrainMean
    xdata_train/=xtrainStd

    xdata_test-=xtrainMean
    xdata_test/=xtrainStd

    ydata_train-=ytrainMean
    ydata_train/=ytrainStd

    ydata_test-=ytrainMean
    ydata_test/=ytrainStd
#    return xdata_train,ydata_train,xdata_test,ydata_test
#最大最小归一化 范围（0，1）
def normalization2(self):
    xtrainMax=np.max(xdata_train,axis=0)
    xtrainMin=np.min(xdata_train,axis=0)
    ytrainMax=np.max(ydata_train,axis=0)
    ytrainMin=np.min(ydata_train,axis=0)
    xdata_train-=xtrainMin
    xdata_train/=(xtrainMax-xtrainMin)

    xdata_test-=xtrainMin
    xdata_test/=(xtrainMax-xtrainMin)

    ydata_train-=ytrainMin
    ydata_train/=(ytrainMax-ytrainMin)

    ydata_test-=ytrainMin
    ydata_test/=(ytrainMax-ytrainMin)
#最大最小归一化 范围（-1，1）
def normalization3(self):
    xtrainMax=np.max(xdata_train,axis=0)
    xtrainMin=np.min(xdata_train,axis=0)
    ytrainMax=np.max(ydata_train,axis=0)
    ytrainMin=np.min(ydata_train,axis=0)
    xdata_train-=xtrainMin
    xdata_train/=(xtrainMax-xtrainMin)
    xdata_train=xdata_train*2-1
    xdata_test-=xtrainMin
    xdata_test/=(xtrainMax-xtrainMin)
    xdata_test=xdata_test*2-1
    ydata_train-=ytrainMin
    ydata_train/=(ytrainMax-ytrainMin)
    ydata_train=ydata_train*2-1
    ydata_test-=ytrainMin
    ydata_test/=(ytrainMax-ytrainMin)
    ydata_test=ydata_test*2-1

#x channel
xf=[i for i in range(165) if i%3==0]
xdata_train1=xdata_train[:,xf]
ydata_train1=ydata_train[:,xf]
xdata_test1=xdata_test[:,xf]
ydata_test1=ydata_test[:,xf]

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

xdata_trainMix=np.concatenate((xdata_train1,xdata_train2,xdata_train3),1)
xdata_testMix=np.concatenate((xdata_test1,xdata_test2,xdata_test3),1)
ydata_trainMix=np.concatenate((ydata_train1,ydata_train2,ydata_train3),1)
ydata_testMix=np.concatenate((ydata_test1,ydata_test2,ydata_test3),1)

#normalization1(xdata_train,ydata_train,xdata_test,ydata_test)
#ytrainMeanMix=np.mean(ydata_trainMix,axis=0)
#ytrainStdMix=np.std(ydata_trainMix,axis=0)
#转置
# xdata_train = np.transpose(x_data[np.arange(train_start, train_end), :])
# xdata_test = np.transpose(x_data[np.arange(test_start, test_end), :])
# ydata_train =np.transpose(y_data[np.arange(train_start, train_end), :])
# ydata_test = np.transpose(y_data[np.arange(test_start, test_end), :])

 # 2.定义节点准备接收数据
# global_step=tf.Variable(0)
# learning_rate=tf.train.exponential_decay(0.1,global_step,100,0.96,staircase=True
#modelBP=BP()

xs1 = tf.placeholder(tf.float32, [None,dimension],name="x_traindata1")
xs2 = tf.placeholder(tf.float32, [None,dimension],name="x_traindata2")
xs3 = tf.placeholder(tf.float32, [None,dimension],name="x_traindata3")
ys= tf.placeholder(tf.float32, [None,dimension*3],name="y_traindata")
#ys2 = tf.placeholder(tf.float32, [None,dimension],name="y_traindata2")
#ys3 = tf.placeholder(tf.float32, [None,dimension],name="y_traindata3")
def LeakyRelu(x, leak=0.2, name="LeakyRelu"):
 with tf.variable_scope(name):
     f1 = 0.5 * (1 + leak)
     f2 = 0.5 * (1 - leak)
     return f1 * x + f2 * tf.abs(x)
#prediction = regression(xs)

def add_layer(wname,bname,inputs, in_size, out_size, activation_function=None):
    with tf.variable_scope("BPNet") :
        weights = tf.get_variable(wname,shape=[in_size,out_size], initializer=tf.contrib.layers.xavier_initializer(), regularizer = tf.contrib.layers.l1_regularizer(0.01))#xavier_initializer()variance_scaling_initializer()
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
w21,h21=add_layer('w21','b21',h11,hidden_layers1,hidden_layers2,activation_function=tf.nn.relu)
w31,h31=add_layer('w31','b31',h21,hidden_layers2,hidden_layers3,activation_function=tf.nn.relu)
w41,h41=add_layer('w41','b41',h31,hidden_layers3,hidden_layers4,activation_function=tf.nn.relu)
w51,h51=add_layer('w51','b51',h41,hidden_layers4,hidden_layers5,activation_function=tf.nn.relu)
w61,prediction1=add_layer('w61','b61',h51,hidden_layers5,dimension)

w12,h12=add_layer('w12','b12',xs2,dimension,hidden_layers1)
w22,h22=add_layer('w22','b22',h12,hidden_layers1,hidden_layers2,activation_function=tf.nn.relu)
w32,h32=add_layer('w32','b32',h22,hidden_layers2,hidden_layers3,activation_function=tf.nn.relu)
w42,h42=add_layer('w42','b42',h32,hidden_layers3,hidden_layers4,activation_function=tf.nn.relu)
w52,h52=add_layer('w52','b52',h42,hidden_layers4,hidden_layers5,activation_function=tf.nn.relu)
w62,prediction2=add_layer('w62','b62',h52,hidden_layers5,dimension)

w13,h13=add_layer('w13','b13',xs3,dimension,hidden_layers1)
w23,h23=add_layer('w23','b23',h13,hidden_layers1,hidden_layers2,activation_function=tf.nn.relu)
w33,h33=add_layer('w33','b33',h23,hidden_layers2,hidden_layers3,activation_function=tf.nn.relu)
w43,prediction3=add_layer('w43','b43',h33,hidden_layers3,dimension)

#
#        w4,h4=add_layer('w4','b4',h3,hidden_layers3,hidden_layers4,activation_function=tf.nn.relu)
#        h4 = tf.nn.dropout(h4,keep_prob)
#        w5,h5=add_layer('w5','b5',h4,hidden_layers4,hidden_layers5,activation_function=tf.nn.relu)
#        h5 = tf.nn.dropout(h5,keep_prob)
#        w6,h6=add_layer('w6','b6',h5,hidden_layers5,hidden_layers6,activation_function=tf.nn.relu)
#        w7,h7=add_layer('w7','b7',h6,hidden_layers6,hidden_layers7,activation_function=tf.nn.relu)
#        w8,prediction=add_layer('w8','b8',h7,hidden_layers7,hidden_layers8)
prediction=tf.concat([prediction1,prediction2,prediction3],1)
ys1,ys2,ys3=tf.split(ys,3,1)
#prediction=tf.concat(1,[prediction1,prediction2,prediction3])
#0.01*tf.nn.l2_loss(w4) +0.01*tf.nn.l2_loss(w5)
thickpre=tf.sqrt(tf.square(prediction1-xs1)+tf.square(prediction2-xs2)+tf.square(prediction3-xs3))
thickreal=tf.sqrt(tf.square(ys1-xs1)+tf.square(ys2-xs2)+tf.square(ys3-xs3))
loss=tf.reduce_sum(tf.reduce_mean(tf.abs(thickpre-thickreal),0))
train_step =tf.train.AdamOptimizer(lrate).minimize(loss)
#accurate=tf.reduce_sum(tf.sign(tf.abs(prediction1-ys1)-1))
#loss=tf.reduce_mean(tf.square(prediction-ys))
def acc(disTest):
    count=0
    for x in range(len(disTest)):
        for i in disTest[x]:
            if abs(i)<1:
                count+=1
    measure=count/np.size(disTest)
    return measure


#image_summary = tf.summary.image
scalar_summary = tf.summary.scalar
#histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter
loss_summary = scalar_summary('loss',loss)
validloss_summary =scalar_summary('validloss',loss)
#acc_summary =scalar_summary('acc',accurate)
#validacc_summary =scalar_summary('validacc',accurate)
trainmerged =merge_summary([loss_summary])
validmerged =merge_summary([validloss_summary])
#def batch(i):
#    start=(i*batch_size)%total
#    end=min(start+batch_size,total)
#    trainx={xs:xdata_train[start:end]}
#    train={xs: xdata_train[start:end],ys:ydata_train[start:end]}

#tf.train.GradientDescentOptimizer(lrate).minimize(loss) # SGD,随机梯度下降
# 使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放
#内存，所以会导致碎片
savetrain = tf.train.Saver(max_to_keep = 5)
if channel:
    test={xs1:xdata_test1,xs2:xdata_test2,xs3:xdata_test3,ys:ydata_testMix}
    testx={xs1:xdata_test1,xs2:xdata_test2,xs3:xdata_test3}
    trainx={xs1:xdata_train1,xs2:xdata_train2,xs3:xdata_train3}
    train={xs1:xdata_train1,xs2:xdata_train2,xs3:xdata_train3,ys:ydata_trainMix}
#else:
#    test={xs:xdata_test,ys: ydata_test}#,keep_prob: 0.5}
#    testx={xs: xdata_test,keep_prob: 0.5}
#    trainx={xs:xdata_train,keep_prob: 0.5}
#    train={xs: xdata_train,ys:ydata_train,keep_prob: 0.5}

gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True


with tf.Session(config = config) as sess:
    # 初始化所有变量
    init = tf.global_variables_initializer()
    sess.run(init)
    writer = SummaryWriter('./logs',sess.graph)
#    costTrain_history=[ ]
#    costTest_history=[ ]
    MinMean=1000000000
    thickMatrixReal=np.ones(shape=(100,0))
    SquareReal=np.square(ydata_testMix-xdata_testMix)
    for j in range(55):
        thickMatrixReal=np.column_stack((thickMatrixReal,np.sqrt(np.add(np.add(SquareReal[:,j],SquareReal[:,j+55]),SquareReal[:,j+110]))))
        np.savetxt('Realthickness.csv', thickMatrixReal, delimiter = ',')

    for i in range(1,steps+1):

         sess.run(train_step, feed_dict=train)
         costTrain=sess.run(loss, feed_dict=train)
#         costTrain_history.append(costTrain)
         costTest=sess.run(loss, feed_dict=test)
#         costTest_history.append(costTest)
         #调用sess.run运行图，生成一步的训练过程数据
         train_summary = sess.run(trainmerged,feed_dict=train)
         valid_summary = sess.run(validmerged,feed_dict=test)
         #调用train_writer的add_summary方法将训练过程以及训练步数保存
         writer.add_summary(train_summary,i)
         writer.add_summary(valid_summary,i)
#                 acc_history.append(acc(distance))
#                 acc_Testhistory.append(acc(Testdistance))
#                 trainAccmerged = merge_summary([acc_summary])
#                 validAccmerged =merge_summary([validacc_summary])
#                 writer.add_summary(trainAccmerged,i)
#                 writer.add_summary(validAccmerged,i)
         if i % 100 == 0:
             thickMatrix=np.ones(shape=(100,0))

             predict=sess.run(prediction,feed_dict=trainx)
             predictest=sess.run(prediction,feed_dict=testx)
             Testdif=sess.run((prediction-ys),feed_dict=test)
             print("difTest:\n",Testdif)
             Square=np.square(predictest-xdata_testMix)
             for j in range(55):
                 thickMatrix=np.column_stack((thickMatrix,np.sqrt(np.add(np.add(Square[:,j],Square[:,j+55]),Square[:,j+110]))))


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
#             print("distanceTrainMean&&Std:\n",dismeanTrain,disStdTrain)
#              if normalized:
#                 predict=sess.run(prediction,feed_dict=trainx)
#                 predictest=sess.run(prediction,feed_dict=testx)
#             print('predictiontrain \n',predict)
#             print('predictionTest:\n',predictest)
             savetrain.save(sess,os.path.join("ckpt1/model.ckpt"),global_step=i)
             if i%2000 ==0:
               np.savetxt('dismean.csv', Savedismean, delimiter = ',')
               np.savetxt('disstd.csv', Savestd, delimiter = ',')
               np.savetxt('original.csv', Savepre, delimiter = ',')
               np.savetxt('thickness.csv', SaveMatrixpre, delimiter = ',')



tmp=np.std(thickMatrixReal,0)
plt.plot(tmp,Savedismean,'ro')
plt.xlabel ( u'55个厚度方差' )
plt.ylabel ( u'55个厚度预测误差均值' )
plt.title ( u'预测准确性与特征规律性关系' )

        #每50次进行一次裁剪
        #saver = tf.train.Saver()
#            checkpoint_path = os.path.join(checkpoint_dir, model_name+'-' + str(counter))
#            savetrain.restore(sess, checkpoint_path)

#    predict=sess.run(prediction,feed_dict={xs:xdata_test})
#如果归一化了，结果得反归一
#    predict=predict*ytrainStd+ytrainMean
#    np.savetxt('predict.csv', predict*ytrainStd+ytrainMean, delimiter = ',')
#    error=(predict-ydata_test)*ytrainStd
#    np.savetxt('error.csv', error, delimiter = ',')
#    saver = tf.train.Saver()
#saver.restore(sess, "/tmp/model.ckpt")
#    costTrain=sess.run(loss, feed_dict=train)
#    print(costTrain,np.min(costTrain_history))
#    plt.plot ( range ( len ( costTrain_history ) ) ,costTrain_history )
#    plt.axis ( [ 0,steps,0,100] )
#    plt.xlabel ( 'training epochs' )
#    plt.ylabel ( 'cost' )
#    plt.title ( 'costTrain history' )
#    plt.show ( )
#    costTest=sess.run(loss, feed_dict=test)
#    print(costTest,np.min(costTest_history))
#    plt.plot ( range ( len ( costTest_history ) ) ,costTest_history )
#    plt.axis ( [ 0,steps,0,1000] )
#    plt.xlabel ( 'training epochs' )
#    plt.ylabel ( 'cost' )
#    plt.title ( 'costTest history' )
#    plt.show ( )
# with tf.Session() as sess:
#     saver.restore(sess,"../model.ckpt")

#     print(sess.run(prediction,feed_dict={xs:xdata_train})