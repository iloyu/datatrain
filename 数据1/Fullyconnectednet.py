#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 17:34:29 2018

@author: iloyu
"""
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import *




class NeuralNet(object):
  def __init__(self,hidden_dims,input_dim=165,output_size,dropout=0,use_batch_norm=False,reg=0.0,weight_scale=1e-2,seed=None,dtype=np.float64):
    '''
    W1:(Features/Dimension,HiddenSize)
    b1:(HiddenSize,)
    W2:(HiddenSize,OutputSize)
    b2:(OutputSize,)
    对于一个L层的神经网络来说，其框架是
    {affine-[batchnorm]-relu-[dropout]}x(L-1)-affine-softmax
    {}重复L-1次[batch norm][dropout]可选
    '''
    self.use_batchnorm=use_batch_norm
    self.use_dropout=dropout>0
    self.reg=reg
    self.num_layers=1+len(hidden_dims)
    self.dtype=dtype
    self.params={}

    in_dim=input_dim
    for i,h_dim in enumerate(hidden_dims):
        self.params['W%d'%(i+1,)]=weight_scale*np.random.randn(in_dim,h_dim)
        self.params['b%d'%(i+1,)]=np.zeros((h_dim,))
        if use_batch_norm:
          self.params["gamma%d"%(i+1,)]=np.ones((h_dim,))
          self.params["beta%d"%(i+1,)]=np.zeros((h_dim,))
        in_dim=h_dim
        """
        开启dropout，需要传递dropout参数字典，保证每层神经元知晓失活概率和当前神经网络的模式状态（训练/测试）
        """
        self.dropout_param={}
        if self.use_dropout:
          self.dropout_param={'mode':'train','p':dropout}
        if seed is not None:
          self.dropout_param['seed']=seed
        """
        开启批量归一化，定义参数列表，跟踪每层的平均值和标准差
        """
        self.bn_params=[]
        if self.use_batchnorm:
          self.bn_params=[{'mode':'train'}for i in range(self.num_layers-1)]

        #调整所有待学习神经网络参数为指定计算精度：np.float64
        for k,v in self.params.items():
          self.params[k]=v.astype(dtype)
  def affine_forward(x,w,b):
    out=None
    reshaped_x=np.reshape(x,(x.shape[0],-1))
    out=reshaped_x.dot(w)+b
    cache=(x,w,b)
  def relu_forward(x):
    out=np.maximum(0,x)
    cahce=x
    return out,cache
  def affine_relu_forward(x,w,b):
    a,fc_cache=affine_forward(x,w,b)
    out,relu_cache=relu_forward(a)
    cache=(fc_cache,relu_cache)
    return out,cache


  def loss(self,X,y=None):
    """
    X:(SampleNum,Dimension)
    y:(SampleNum,Dimension)
    reg:regularization strength
    训练模式下：
    loss函数目标输出一个损失值和梯度字典
    其中存有（W，bIAS，gamma，bETA）
    测试模式下
    loss给出得分
    """
    X=X.astype(self.dtype)
    mode='test'if y is None else 'train'
    if self.dropout_param is not None:
      self.dropout_param['mode']=mode
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param['mode']=mode
    scores=None
    fc_mix_cache={}
    if self.use_dropout:
      dp_cache={}
    out=X
    for i in range(self.num_layers-1):
      w,b=self.params["W%d"%(i+1,)],self.params["b%d"%(i+1,)]
      if self.use_batchnorm:
        gamma=self.params["gamma%d"%(i+1,)]
        beta=self.params["beta%d"%(i+1,)]
        out,fc_mix_cache[i]=affine
    W1,b1=self.params['W1'],self.params['b1']
    W2,b2=self.params['W2'],self.params['b2']
    W3,b3=self.params['W3'],self.params['b3']
    N,D=X.shape
    predic=None
    z1=X.dot(W1)+b1
    a1=np.maximum(0,z1)
    z2=a1.dot(W2)+b2
    a2=np.maximum(0,z2)
    predic=a2.dot(W3)+b3
    if y is None:
      return predic
    loss=None
#    f=predic-np.max(predic,axis=1,keepdims=True)
    probs=np.abs(predic-y)
    probs/=np.sum(np.abs(predic-y),axis=1,keepdims=True)
    f=np.mean(probs)
    loss=f+0.5*reg*(np.sum(W1*W1)+np.sum(W2*W2)+np.sum(W3*W3))
    grads={}
    dpredic=(predic-y)
    grads['W3']=np.dot(a2.T,dpredic)
    grads['b3']=np.sum(dpredic,axis=0)
    dhidden2=np.dot(dpredic,W3.T)
    dhidden2[a2<=0]=0.01*dhidden2[a2<=0]
    grads['W2']=np.dot(a1.T,dhidden2)
    grads['b2']=np.sum(dhidden2,axis=0)
    dhidden1=np.dot(dhidden2,W2.T)
    dhidden1[a1<=0]=0.01*dhidden1[a1<=0]
    grads['W1']=np.dot(X.T,dhidden1)
    grads['b1']=np.sum(dhidden1,axis=0)

    grads['W3']+=reg*W3
    grads['W2']+=reg*W2
    grads['W1']+=reg*W1
    return loss,grads

  def train(self,X,y,X_val,y_val,learning_rate=1e-3,learning_rate_decay=0.95,reg=1e-5,num_iters=100,batch_size=200,verbose=False):
    """
    SGD训练，
    X：(N,D)
    y:(N,D)
    num_iters:优化迭代的次数
    batch_size:每次训练使用的训练样本数
    verbose:是否打印迭代过程
    """
    num_train=X.shape[0]
    iterations_per_epoch=max(num_train/batch_size,1)
    loss_history=[]
    train_acc_history=[]
    val_acc_history=[]
    for it in range(num_iters):
      X_batch=None
      y_batch=None
      indices=np.random.choice(num_train,batch_size,replace=True)
      X_batch=X[indices]
      y_batch=y[indices]
      loss,grads=self.loss(X_batch,y=y_batch,reg=reg)
      loss_history.append(loss)
      self.params['W1']-=learning_rate*grads['W1']
      self.params['b1']-=learning_rate*grads['b1']
      self.params['W2']-=learning_rate*grads['W2']
      self.params['b2']-=learning_rate*grads['b2']
      self.params['W3']-=learning_rate*grads['W3']
      self.params['b3']-=learning_rate*grads['b3']
      if verbose and it%100==0:
        print('iteration %d/%d:loss %f'%(it,num_iters,loss))
      if it %iterations_per_epoch==0:
         train_acc = (np.abs(self.predict(X_batch) - y_batch)<1).mean()
         val_acc = (np.abs(self.predict(X_val) - y_val)<1).mean()
         train_acc_history.append(train_acc)
         val_acc_history.append(val_acc)
         learning_rate*=learning_rate_decay
    return {
        'loss_history':loss_history,
        'train_acc_history':train_acc_history,
        'val_acc_history':val_acc_history
        }

  def predict(self,X):
    y_pred=None
    W1,b1=self.params['W1'],self.params['b1']
    W2,b2=self.params['W2'],self.params['b2']
    W3,b3=self.params['W3'],self.params['b3']
    hidden1=np.maximum(0,np.dot(X,W1)+b1)
    hidden2=np.maximum(0,np.dot(hidden1,W2)+b2)
    y_pred=np.dot(hidden2,W3)+b3
    return y_pred


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

#
#xs = tf.placeholder(tf.float32, [None,dimension],name="x_traindata")
#ys= tf.placeholder(tf.float32, [None,dimension],name="y_traindata")
#def LeakyRelu(x, leak=0.2, name="LeakyRelu"):
# with tf.variable_scope(name):
#     f1 = 0.5 * (1 + leak)
#     f2 = 0.5 * (1 - leak)
#     return f1 * x + f2 * tf.abs(x)
##prediction = regression(xs)
#
#def add_layer(wname,bname,inputs, in_size, out_size, activation_function=None):
#    with tf.variable_scope("BPNet") :
#        weights = tf.get_variable(wname,shape=[in_size,out_size], initializer=tf.contrib.layers.xavier_initializer(), regularizer = tf.contrib.layers.l1_regularizer(0.01))#xavier_initializer()variance_scaling_initializer()
#        b = tf.get_variable(bname, shape=[1,out_size], initializer=tf.zeros_initializer())
#        Wx_plus_b = tf.matmul(inputs, weights) + b
#        if activation_function is None:
#            outputs = Wx_plus_b
#        else:
#            outputs = activation_function(Wx_plus_b)
#       # tf.histogram_summary(wname + '/outputs', outputs)#变量跟踪
#        return weights,outputs
#
### 构建输入层到隐藏层,假设隐藏层有 hidden_layers 个神经元
#
#w11,h11=add_layer('w11','b11',xs,dimension,hidden_layers1)
#w21,h21=add_layer('w21','b21',h11,hidden_layers1,hidden_layers2,activation_function=LeakyRelu)
#w31,h31=add_layer('w31','b31',h21,hidden_layers2,hidden_layers3,activation_function=tf.nn.relu)
#w41,prediction=add_layer('w41','b41',h31,hidden_layers3,hidden_layers4)
#
##w12,h12=add_layer('w12','b12',xs2,dimension,hidden_layers1)
##w22,h22=add_layer('w22','b22',h12,hidden_layers1,hidden_layers2,activation_function=tf.nn.relu)
##w32,h32=add_layer('w32','b32',h22,hidden_layers2,hidden_layers3,activation_function=tf.nn.relu)
##w42,prediction2=add_layer('w42','b42',h32,hidden_layers3,hidden_layers4)
##
##w13,h13=add_layer('w13','b13',xs3,dimension,hidden_layers1)
##w23,h23=add_layer('w23','b23',h13,hidden_layers1,hidden_layers2,activation_function=tf.nn.relu)
##w33,h33=add_layer('w33','b33',h23,hidden_layers2,hidden_layers3,activation_function=tf.nn.relu)
##w43,prediction3=add_layer('w43','b43',h33,hidden_layers3,hidden_layers4)
#
##
##        w4,h4=add_layer('w4','b4',h3,hidden_layers3,hidden_layers4,activation_function=tf.nn.relu)
##        h4 = tf.nn.dropout(h4,keep_prob)
##        w5,h5=add_layer('w5','b5',h4,hidden_layers4,hidden_layers5,activation_function=tf.nn.relu)
##        h5 = tf.nn.dropout(h5,keep_prob)
##        w6,h6=add_layer('w6','b6',h5,hidden_layers5,hidden_layers6,activation_function=tf.nn.relu)
##        w7,h7=add_layer('w7','b7',h6,hidden_layers6,hidden_layers7,activation_function=tf.nn.relu)
##        w8,prediction=add_layer('w8','b8',h7,hidden_layers7,hidden_layers8)
#
##prediction=tf.concat(1,[prediction1,prediction2,prediction3])
##0.01*tf.nn.l2_loss(w4) +0.01*tf.nn.l2_loss(w5)
#loss=tf.reduce_mean(tf.reduce_sum(tf.abs(prediction-ys),axis=1))
#train_step =tf.train.AdamOptimizer(lrate).minimize(loss)
##accurate=tf.reduce_sum(tf.sign(tf.abs(prediction1-ys1)-1))
##loss=tf.reduce_mean(tf.square(prediction-ys))
#def acc(disTest):
#    count=0
#    for x in range(len(disTest)):
#        for i in disTest[x]:
#            if abs(i)<1:
#                count+=1
#    measure=count/np.size(disTest)
#    return measure
#
#
#scalar_summary = tf.summary.scalar
#
#merge_summary = tf.summary.merge
#SummaryWriter = tf.summary.FileWriter
#loss_summary = scalar_summary('loss',loss)
#validloss_summary =scalar_summary('validloss',loss)
##acc_summary =scalar_summary('acc',accurate)
##validacc_summary =scalar_summary('validacc',accurate)
#trainmerged =merge_summary([loss_summary])
#validmerged =merge_summary([validloss_summary])
#
##tf.train.GradientDescentOptimizer(lrate).minimize(loss) # SGD,随机梯度下降
## 使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放
##内存，所以会导致碎片
#savetrain = tf.train.Saver(max_to_keep = 5)
##if channel!=0:
##    pass
###    test={xs1:xdata_test1,xs2:xdata_test2,xs3:xdata_test3,ys:ydata_testMix}
###    testx={xs1:xdata_test1,xs2:xdata_test2,xs3:xdata_test3}
###    trainx={xs1:xdata_train1,xs2:xdata_train2,xs3:xdata_train3}
###    train={xs1:xdata_train1,xs2:xdata_train2,xs3:xdata_train3,ys:ydata_trainMix}
##else:
#test={xs:xdata_test,ys: ydata_test}#,keep_prob: 0.5}
#testx={xs: xdata_test}#,keep_prob: 0.5}
#trainx={xs:xdata_train}#,keep_prob: 0.5}
#train={xs: xdata_train,ys:ydata_train}#,keep_prob: 0.5}
#
#gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
#config = tf.ConfigProto(gpu_options=gpu_options)
#config.gpu_options.allow_growth = True
#
#
#with tf.Session(config = config) as sess:
#    # 初始化所有变量
#    init = tf.global_variables_initializer()
#    sess.run(init)
#    writer = SummaryWriter('./log2',sess.graph)
##    costTrain_history=[ ]
##    costTest_history=[ ]
#    MinMean=1000000000
#    thickMatrixReal=np.ones(shape=(num_valid,0))
#    SquareReal=np.square(ydata_test-xdata_test)
#    for j in range(0,163,3):
#        thickMatrixReal=np.column_stack((thickMatrixReal,np.sqrt(np.add(np.add(SquareReal[:,j],SquareReal[:,j+1]),SquareReal[:,j+2]))))
#
#    for i in range(1,steps+1):
#
#         sess.run(train_step, feed_dict=train)
#         costTrain=sess.run(loss, feed_dict=train)
##         costTrain_history.append(costTrain)
#         costTest=sess.run(loss, feed_dict=test)
##         costTest_history.append(costTest)
#         #调用sess.run运行图，生成一步的训练过程数据
#         train_summary = sess.run(trainmerged,feed_dict=train)
#         valid_summary = sess.run(validmerged,feed_dict=test)
#         #调用train_writer的add_summary方法将训练过程以及训练步数保存
#         writer.add_summary(train_summary,i)
#         writer.add_summary(valid_summary,i)
##                 acc_history.append(acc(distance))
##                 acc_Testhistory.append(acc(Testdistance))
##                 trainAccmerged = merge_summary([acc_summary])
##                 validAccmerged =merge_summary([validacc_summary])
##                 writer.add_summary(trainAccmerged,i)
##                 writer.add_summary(validAccmerged,i)
#         if i % 100 == 0:
#             thickMatrix=np.ones(shape=(num_valid,0))
#
#             predict=sess.run(prediction,feed_dict=trainx)
#             predictest=sess.run(prediction,feed_dict=testx)
#             Testdif=sess.run((prediction-ys),feed_dict=test)
#             print("difTest:\n",Testdif)
#             Square=np.square(predictest-xdata_test)
#             for j in range(0,163,3):
#                 thickMatrix=np.column_stack((thickMatrix,np.sqrt(np.add(np.add(Square[:,j],Square[:,j+1]),Square[:,j+2]))))
#
#
#             dismean=np.mean(abs(thickMatrixReal-thickMatrix),0)
##             disStdTrain=np.std(abs(distance),0)
##             dismean=np.mean(abs(Testdistance),0)
#             disStd=np.std(abs(thickMatrixReal-thickMatrix),0)
#             disMeantmp=np.mean(dismean)
#             if disMeantmp<MinMean:
#               MinMean=disMeantmp
#               Savedismean=dismean
#               Savestd=disStd
#               Savepre=predictest
#             print("distanceTestMean&&Std:\n",dismean,disStd,"\nTestMean",disMeantmp)
##             print("distanceTrainMean&&Std:\n",dismeanTrain,disStdTrain)
##              if normalized:
##                 predict=sess.run(prediction,feed_dict=trainx)
##                 predictest=sess.run(prediction,feed_dict=testx)
##             print('predictiontrain \n',predict)
##             print('predictionTest:\n',predictest)
#             savetrain.save(sess,os.path.join("ckpt1/model.ckpt"),global_step=i)
#             if i%2000 ==0:
#               np.savetxt('dismean165.csv', Savedismean, delimiter = ',')
#               np.savetxt('disstd165.csv', Savestd, delimiter = ',')
#               np.savetxt('original165.csv', Savepre, delimiter = ',')
#
#tmp=np.std(thickMatrixReal,0)
#plt.plot(tmp,Savedismean,'ro')
#plt.xlabel ( u'55个厚度方差' )
#plt.ylabel ( u'55个厚度预测误差均值' )
#plt.title ( u'预测准确性与特征规律性关系' )
#        #每50次进行一次裁剪
#        #saver = tf.train.Saver()
##            checkpoint_path = os.path.join(checkpoint_dir, model_name+'-' + str(counter))
##            savetrain.restore(sess, checkpoint_path)
#
##    predict=sess.run(prediction,feed_dict={xs:xdata_test})
##如果归一化了，结果得反归一
##    predict=predict*ytrainStd+ytrainMean
##    np.savetxt('predict.csv', predict*ytrainStd+ytrainMean, delimiter = ',')
##    error=(predict-ydata_test)*ytrainStd
##    np.savetxt('error.csv', error, delimiter = ',')
##    saver = tf.train.Saver()
##saver.restore(sess, "/tmp/model.ckpt")
##    costTrain=sess.run(loss, feed_dict=train)
##    print(costTrain,np.min(costTrain_history))
##    plt.plot ( range ( len ( costTrain_history ) ) ,costTrain_history )
##    plt.axis ( [ 0,steps,0,100] )
##    plt.xlabel ( 'training epochs' )
##    plt.ylabel ( 'cost' )
##    plt.title ( 'costTrain history' )
##    plt.show ( )
##    costTest=sess.run(loss, feed_dict=test)
##    print(costTest,np.min(costTest_history))
##    plt.plot ( range ( len ( costTest_history ) ) ,costTest_history )
##    plt.axis ( [ 0,steps,0,1000] )
##    plt.xlabel ( 'training epochs' )
##    plt.ylabel ( 'cost' )
##    plt.title ( 'costTest history' )
##    plt.show ( )
## with tf.Session() as sess:
##     saver.restore(sess,"../model.ckpt")
#
##     print(sess.run(prediction,feed_dict={xs:xdata_train})