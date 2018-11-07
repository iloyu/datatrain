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
class BP(object):
    def __init__(self):
        tf.reset_default_graph()
    #self.merged_summary=tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,BPNet))
        self.Numsize=500#样本数
        self.feature=165 #特征数
        self.lrate=0.001 #学习速率
        self.batch_size =0#100#训练batch的大小
        self.normalized=1#是否归一化
        self.steps=20000#迭代次数
        self.dimension=55#属性
        self.channel=3 #x,y,z
        self.total=400#训练样本数


        self.train_start = 0
        self.train_end = int(np.floor(0.8*self.Numsize))
        self.test_start = self.train_end
        self.test_end = self.Numsize

        self.hidden_layers1=self.dimension
        self.hidden_layers2=1024
        self.hidden_layers3=1024
        self.hidden_layers4=self.dimension#1024#1024
        self.hidden_layers5=2048
        self.hidden_layers6=512
        self.hidden_layers7=256
        self.hidden_layers8=self.dimension

    def Data(self):
        inputX= pd.read_csv('xDataR2.txt',header=None,sep=',')
        inputY=pd.read_csv('yDataR2.txt',header=None,sep=',')
        self.x_data =inputX.values#[:,0:dimension]
        self.y_data=inputY.values#[:,0:dimension]
    def divideData(self):
        self.xdata_train =self.x_data[np.arange(self.train_start, self.train_end),:]
        self.xdata_test = self.x_data[np.arange(self.test_start, self.test_end), :]
        self.ydata_train =self.y_data[np.arange(self.train_start,self.train_end), :]
        self.ydata_test = self.y_data[np.arange(self.test_start,self.test_end), :]

    # print(xdata_test,xdata_test.shape)
    # print(ydata_test,ydata_test.shape)
    #均值方差归一化，范围（-1，1）
    def normalization1(self):

        xtrainMean=np.mean(self.xdata_train,axis=0)
        xtrainStd=np.std(self.xdata_train,axis=0)
        ytrainMean=np.mean(self.ydata_train,axis=0)
        ytrainStd=np.std(self.ydata_train,axis=0)
        self.xdata_train-=xtrainMean
        self.xdata_train/=xtrainStd

        self.xdata_test-=xtrainMean
        self.xdata_test/=xtrainStd

        self.ydata_train-=ytrainMean
        self.ydata_train/=ytrainStd

        self.ydata_test-=ytrainMean
        self.ydata_test/=ytrainStd
        #最大最小归一化 范围（0，1）
    def normalization2(self):
        xtrainMax=np.max(self.xdata_train,axis=0)
        xtrainMin=np.min(self.xdata_train,axis=0)
        ytrainMax=np.max(self.ydata_train,axis=0)
        ytrainMin=np.min(self.ydata_train,axis=0)
        self.xdata_train-=xtrainMin
        self.xdata_train/=(xtrainMax-xtrainMin)

        self.xdata_test-=xtrainMin
        self.xdata_test/=(xtrainMax-xtrainMin)

        self.ydata_train-=ytrainMin
        self.ydata_train/=(ytrainMax-ytrainMin)

        self.ydata_test-=ytrainMin
        self.ydata_test/=(ytrainMax-ytrainMin)
        #最大最小归一化 范围（-1，1）
    def normalization3(self):
        xtrainMax=np.max(self.xdata_train,axis=0)
        xtrainMin=np.min(self.xdata_train,axis=0)
        ytrainMax=np.max(self.ydata_train,axis=0)
        ytrainMin=np.min(self.ydata_train,axis=0)
        self.xdata_train-=xtrainMin
        self.xdata_train/=(xtrainMax-xtrainMin)
        self.xdata_train=self.xdata_train*2-1
        self.xdata_test-=xtrainMin
        self.xdata_test/=(xtrainMax-xtrainMin)
        self.xdata_test=self.xdata_test*2-1
        self.ydata_train-=ytrainMin
        self.ydata_train/=(ytrainMax-ytrainMin)
        self.ydata_train=self.ydata_train*2-1
        self.ydata_test-=ytrainMin
        self.ydata_test/=(ytrainMax-ytrainMin)
        self.ydata_test=self.ydata_test*2-1
    def ThreeChannel(self):
        #x channel
        xf=[i for i in range(165) if i%3==0]
        self.xdata_train1=self.xdata_train[:,xf]
        self.ydata_train1=self.ydata_train[:,xf]
        self.xdata_test1=self.xdata_test[:,xf]
        self.ydata_test1=self.ydata_test[:,xf]

        #y channel
        xf=[i for i in range(165) if i%3==1]
        self.xdata_train2=self.xdata_train[:,xf]
        self.ydata_train2=self.ydata_train[:,xf]
        self.xdata_test2=self.xdata_test[:,xf]
        self.ydata_test2=self.ydata_test[:,xf]
        #z channel
        xf=[i for i in range(165) if i%3==2]
        self.xdata_train3=self.xdata_train[:,xf]
        self.ydata_train3=self.ydata_train[:,xf]
        self.xdata_test3=self.xdata_test[:,xf]
        self.ydata_test3=self.ydata_test[:,xf]

        self.xdata_trainMix=np.concatenate((self.xdata_train1,self.xdata_train2,self.xdata_train3),1)
        self.xdata_testMix=np.concatenate((self.xdata_test1,self.xdata_test2,self.xdata_test3),1)
        self.ydata_trainMix=np.concatenate((self.ydata_train1,self.ydata_train2,self.ydata_train3),1)
        self.ydata_testMix=np.concatenate((self.ydata_test1,self.ydata_test2,self.ydata_test3),1)
def normalization3channel(self):

        xtrainMean=np.mean(self.xdata_trainMix,axis=0)
        xtrainStd=np.std(self.xdata_trainMix,axis=0)
        ytrainMean=np.mean(self.ydata_trainMix,axis=0)
        ytrainStd=np.std(self.ydata_trainMix,axis=0)
        self.xdata_trainN=self.xdata_trainMix-xtrainMean
        self.xdata_trainN/=xtrainStd

        self.xdata_testN=self.xdata_testMix-xtrainMean
        self.xdata_testN/=xtrainStd

        self.ydata_trainN=self.ydata_trainMix-ytrainMean
        self.ydata_trainN/=ytrainStd

        self.ydata_testN=self.ydata_testMix-ytrainMean
        self.ydata_testN/=ytrainStd

    #转置
    # xdata_train = np.transpose(x_data[np.arange(train_start, train_end), :])
    # xdata_test = np.transpose(x_data[np.arange(test_start, test_end), :])
    # ydata_train =np.transpose(y_data[np.arange(train_start, train_end), :])
    # ydata_test = np.transpose(y_data[np.arange(test_start, test_end), :])

     # 2.定义节点准备接收数据
    # global_step=tf.Variable(0)
    # learning_rate=tf.train.exponential_decay(0.1,global_step,100,0.96,staircase=True
    #modelBP=BP()
    def inputXY(self):
       self.xs1 = tf.placeholder(tf.float32, [None,self.dimension],name="x_traindata1")
       self.xs2 = tf.placeholder(tf.float32, [None,self.dimension],name="x_traindata2")
       self.xs3 = tf.placeholder(tf.float32, [None,self.dimension],name="x_traindata3")
       self.ys= tf.placeholder(tf.float32, [None,self.dimension*3],name="y_traindata")


    def LeakyRelu(x, leak=0.2, name="LeakyRelu"):
         with tf.variable_scope(name):
             f1 = 0.5 * (1 + leak)
             f2 = 0.5 * (1 - leak)
             return f1 * x + f2 * tf.abs(x)
    #prediction = regression(xs)
    def add_layer(self,wname,bname,inputs, in_size, out_size, activation_function=None):
        with tf.variable_scope("BPNet") :
            weights = tf.get_variable(wname,shape=[in_size,out_size], initializer=tf.contrib.layers.xavier_initializer(), regularizer = tf.contrib.layers.l1_regularizer(0.01))#variance_scaling_initializer()
            b = tf.get_variable(bname, shape=[1,out_size], initializer=tf.zeros_initializer())
            Wx_plus_b = tf.matmul(inputs, weights) + b
            if activation_function is None:
                outputs = Wx_plus_b
            else:
                outputs = activation_function(Wx_plus_b)
           # tf.histogram_summary(wname + '/outputs', outputs)#变量跟踪
            return weights,outputs

    ## 构建输入层到隐藏层,假设隐藏层有 hidden_layers 个神经元
    def train_net(self):
       #self.keep_prob = tf.placeholder("float")
        w11,h11=self.add_layer('w11','b11',self.xs1,self.dimension,self.hidden_layers1)
        w21,h21=self.add_layer('w21','b21',h11,self.hidden_layers1,self.hidden_layers2,activation_function=tf.nn.relu)
        w31,h31=self.add_layer('w31','b31',h21,self.hidden_layers2,self.hidden_layers3,activation_function=tf.nn.relu)
        w41,self.prediction1=self.add_layer('w41','b41',h31,self.hidden_layers3,self.hidden_layers4)

        w12,h12=self.add_layer('w12','b12',self.xs2,self.dimension,self.hidden_layers1)
        w22,h22=self.add_layer('w22','b22',h12,self.hidden_layers1,self.hidden_layers2,activation_function=tf.nn.relu)
        w32,h32=self.add_layer('w32','b32',h22,self.hidden_layers2,self.hidden_layers3,activation_function=tf.nn.relu)
        w42,self.prediction2=self.add_layer('w42','b42',h32,self.hidden_layers3,self.hidden_layers4)

        w13,h13=self.add_layer('w13','b13',self.xs3,self.dimension,self.hidden_layers1)
        w23,h23=self.add_layer('w23','b23',h13,self.hidden_layers1,self.hidden_layers2,activation_function=tf.nn.relu)
        w33,h33=self.add_layer('w33','b33',h23,self.hidden_layers2,self.hidden_layers3,activation_function=tf.nn.relu)
        w43,self.prediction3=self.add_layer('w43','b43',h33,self.hidden_layers3,self.hidden_layers4)
# w4,h4=self.add_layer('w4','b4',h3,self.hidden_layers3,self.hidden_layers4,activation_function=tf.nn.relu)
#        h4 = tf.nn.dropout(h4,self.keep_prob)
#        w5,h5=self.add_layer('w5','b5',h4,self.hidden_layers4,self.hidden_layers5,activation_function=tf.nn.relu)
#        h5 = tf.nn.dropout(h5,self.keep_prob)
#        w6,h6=self.add_layer('w6','b6',h5,self.hidden_layers5,self.hidden_layers6,activation_function=tf.nn.relu)
#        w7,h7=self.add_layer('w7','b7',h6,self.hidden_layers6,self.hidden_layers7,activation_function=tf.nn.relu)
#        w8,self.prediction=self.add_layer('w8','b8',h7,self.hidden_layers7,self.hidden_layers8)
#        self.sum=tf.square(self.prediction1-self.ys[0,:])+tf.square(self.prediction2-self.ys[1,:])+tf.square(self.prediction3-self.ys[2,:])#0.01*tf.nn.l2_loss(w4) +0.01*tf.nn.l2_loss(w5)
        self.prediction=tf.concat((self.prediction1,self.prediction2,self.prediction3),1)
        self.loss=tf.reduce_sum(tf.square(self.prediction1-self.ys[0,:])+tf.square(self.prediction2-self.ys[1,:])+tf.square(self.prediction3-self.ys[2,:]))
        self.train_step =tf.train.AdamOptimizer(self.lrate).minimize(self.loss)
        self.accurate=tf.reduce_sum(tf.sign(tf.abs(self.prediction1-self.ys[0,:])-1))
    #loss=tf.reduce_mean(tf.square(prediction-ys))
    def acc(self,disTest):
        count=0
        for x in range(len(disTest)):
            for i in disTest[x]:
                if abs(i)<1:
                    count+=1
        measure=count/np.size(disTest)
        return measure

    def visualize(self):
        #image_summary = tf.summary.image
        self.scalar_summary = tf.summary.scalar
        #histogram_summary = tf.summary.histogram
        self.merge_summary = tf.summary.merge
        self.SummaryWriter = tf.summary.FileWriter
        self.loss_summary = self.scalar_summary('loss',self.loss)
        self.validloss_summary =self.scalar_summary('validloss',self.loss)
        self.acc_summary =self.scalar_summary('acc',self.accurate)
        self.validacc_summary =self. scalar_summary('validacc',self.accurate)
        self.trainmerged =self.merge_summary([self.loss_summary,self.acc_summary])
        self.validmerged =self.merge_summary([self.validloss_summary,self.validacc_summary])
    def batch(self,i):
        start=(i*self.batch_size)%self.total
        end=min(start+self.batch_size,self.total)
        self.trainx={self.xs:self.xdata_train[start:end]}
        self.train={self.xs: self.xdata_train[start:end],self.ys:self.ydata_train[start:end]}
    def trainConfig(self):
        #tf.train.GradientDescentOptimizer(lrate).minimize(loss) # SGD,随机梯度下降
        # 使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放
        #内存，所以会导致碎片
        self.savetrain = tf.train.Saver(max_to_keep = 5)
        if self.channel:
            self.test={self.xs:self.xdata_testMix,self.ys: self.ydata_testMix}#,self.keep_prob: 0.5
            self.testx={self.xs: self.xdata_testMix}#,self.keep_prob: 0.5
            self.trainx={self.xs:self.xdata_trainMix}#,self.keep_prob: 0.5
            self.train={self.xs: self.xdata_trainMix,self.ys:self.ydata_trainMix}#,self.keep_prob: 0.5
        else:
            self.test={self.xs:self.xdata_test,self.ys: self.ydata_test,self.keep_prob: 0.5}
            self.testx={self.xs: self.xdata_test,self.keep_prob: 0.5}
            self.trainx={self.xs:self.xdata_train,self.keep_prob: 0.5}
            self.train={self.xs: self.xdata_train,self.ys:self.ydata_train,self.keep_prob: 0.5}
    def run(self):
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        self.config = tf.ConfigProto(gpu_options=gpu_options)
        self.config.gpu_options.allow_growth = True
        self.visualize()
        with tf.Session(config = self.config) as sess:
            # 初始化所有变量
            init = tf.global_variables_initializer()
            sess.run(init)
            writer = self.SummaryWriter('./logs',sess.graph)
            costTrain_history=[ ]
            costTest_history=[ ]
            acc_history=[ ]
            acc_Testhistory=[ ]
            for i in range(1,self.steps+1):

                 sess.run(self.train_step, feed_dict=self.train)
                 costTrain=sess.run(self.loss, feed_dict=self.train)
                 costTrain_history.append(costTrain)
                 costTest=sess.run(self.loss, feed_dict=self.test)
                 costTest_history.append(costTest)
                 #调用sess.run运行图，生成一步的训练过程数据
                 train_summary = sess.run(self.trainmerged,feed_dict=self.train)
                 valid_summary = sess.run(self.validmerged,feed_dict=self.test)
                 #调用train_writer的add_summary方法将训练过程以及训练步数保存
                 writer.add_summary(train_summary,i)
                 writer.add_summary(valid_summary,i)
                 if self.normalized:
                     distance=sess.run((self.prediction-self.ys)*self.ytrainStd,feed_dict=self.train)
                     Testdistance=sess.run((self.prediction-self.ys)*self.ytrainStd,feed_dict=self.test)
                 else:
                     distance=sess.run((self.prediction1-self.ys[0,:]),feed_dict=self.train)
                     Testdistance=sess.run((self.prediction1-self.ys[0,:]),feed_dict=self.test)


#                 acc_history.append(self.acc(distance))
#                 acc_Testhistory.append(self.acc(Testdistance))
#                 trainAccmerged = self.merge_summary([acc_summary])
#                 validAccmerged =self.merge_summary([validacc_summary])
#                 writer.add_summary(trainAccmerged,i)
#                 writer.add_summary(validAccmerged,i)
                 if i % 100 == 0:
                     print("distanceTrain:\n",distance)
                     print("distanceTest:\n",Testdistance)
                     predict=sess.run(self.prediction1,feed_dict=self.trainx)
                     predictest=sess.run(self.prediction1,feed_dict=self.testx)
                     if self.normalized:
                         predict=sess.run(self.prediction1*self.ytrainStd+self.ytrainMean,feed_dict=self.trainx)
                         predictest=sess.run(self.prediction1*self.ytrainStd+self.ytrainMean,feed_dict=self.testx)
                     print('predictiontrain \n',predict)
                     print('predictionTest:\n',predictest)
                     self.savetrain.save(sess,os.path.join("ckpt/model.ckpt"),global_step=i)
    def main(self):
      self.Data()
      self.divideData()
      self.ThreeChannel()
      self.inputXY()
      self.train_net()
      self.visualize()
      self.trainConfig()
      self.run()

BPnet=BP()
BPnet.main()

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