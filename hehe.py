#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import division
from __future__ import print_function  
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
# import seaborn as sns
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import tushare as ts

def getData(id,start,end,num,flag):
    df = ts.get_hist_data(id,start,end)
    #df = (df-np.sum(df)/len(df))/(np.std(df))
    if(flag=="true"):
        df = df[1:num]
    else:
        df = df[:num]
    df1 = np.array(df)
    #df2 = np.array(df.index)
    
    ##df = df.T
    x = []
    for i in range(len(df1)):
        #temp = np.append(df2[i],df1[i])
        temp = df1[i]
        newresult = []
        for item in temp:
            newresult.append(item)
        x.append(newresult)
    x.reverse()
    return x


def getDataR(id,start,end,num):
    df = ts.get_hist_data(id,start,end)
    df1 = np.array(df)
    x = []
    for i in range(len(df1)):
        temp = df1[i]
        newresult = []
        for item in temp:
            newresult.append(item)
        x.append(newresult)
    
    P=df['close']
    #实际上是没有end那一天的数据，这里是预测未来一天相对于现在的收盘价
    templist=(P.shift(1)-P)/P
    templist = templist[:num]
    templist = np.array(templist)
    templist = templist.tolist()
    templist.reverse()
    tempDATA = []
    for i in range(len(templist)):
        if((i+1)%10!=0):
            pass
        else:
            if(templist[i]>0):
                #tempDATA.append(templist[i])
                tempDATA.append([1,0,0])
            elif(templist[i]<=0):
                #tempDATA.append(templist[i])
                tempDATA.append([0,1,0])
            else:
                #tempDATA.append(templist[i])
                tempDATA.append([0,0,1])
            
    y=tempDATA
    return y

#df_sh = ts.get_sz50s()['code']
df_sh =["600583"]
fac = []
ret = []
facT = []
retT = []
predFAC = []
for ishare in df_sh:
    #取最近的260天数据
    newfac = getData(ishare,'2008-07-22','2016-08-01',601,"true")
    newret = getDataR(ishare,'2008-07-22','2016-08-01',601)
    #fac.append(newfac)
    for i in range(len(newfac)):
        fac.append(newfac[i])
    for i in range(len(newret)):
        ret.append(newret[i])
    
    newfacT = getData(ishare,'2016-08-01','2017-01-19',101,"true")
    newretT = getDataR(ishare,'2016-08-01','2017-01-19',101)
    #fac.append(newfac)
    for i in range(len(newfacT)):
        facT.append(newfacT[i])
    for i in range(len(newretT)):
        retT.append(newretT[i])
    
    newpredFAC = getData(ishare,'2016-08-01','2017-01-20',11,"false")
    for i in range(len(newpredFAC)):
        predFAC.append(newpredFAC[i])

fac = np.array(fac)
ret = np.array(ret)
meanfac = np.sum(fac, axis=0)/len(fac)
stdfac = np.std(fac, axis=0)
fac = (fac-meanfac)/stdfac

facT = np.array(facT)
retT = np.array(retT)
facT = (facT-meanfac)/stdfac


newf = []
newfa = []
for i in range(len(fac)):
    if((i+1)%10!=0):
        newf.append(fac[i])
    else:
        newf.append(fac[i])
        newfa.append(newf)
        newf = []
fac = np.array(newfa)
newfT = []
newfaT = []
for i in range(len(facT)):
    if((i+1)%10!=0):
        newfT.append(facT[i])
    else:
        newfT.append(facT[i])
        newfaT.append(newfT)
        newfT = []
facT = np.array(newfaT)

predFAC = (predFAC-meanfac)/stdfac


learning_rate = 0.001
batch_size = 10
print(int(fac.shape[0]))
training_iters = int(fac.shape[0]/batch_size)
display_step = 10

# Network Parameters
n_input = 14
n_steps = 10
n_hidden = 1024
n_classes = 3
dropout = 0.8
# tf Graph input
x = tf.placeholder('float',[None, n_steps, n_input])
y = tf.placeholder('float',[None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


def CNN_Net_five(x,weights,biases,dropout=0.8,m=1):
    
    x = tf.reshape(x, shape=[-1,10,14,1])
    
    # 卷积层1
    x = tf.nn.conv2d(x, weights['wc1'], strides=[1,m,m,1],padding='SAME')
    x = tf.nn.bias_add(x,biases['bc1'])
    x = tf.nn.relu(x)
    
    # 卷积层2 
    x = tf.nn.conv2d(x, weights['wc2'], strides=[1,m,m,1],padding='SAME')
    x = tf.nn.bias_add(x,biases['bc2'])
    x = tf.nn.relu(x)
    
    # 卷积层3 
    x = tf.nn.conv2d(x, weights['wc3'], strides=[1,m,m,1],padding='SAME')
    x = tf.nn.bias_add(x,biases['bc3'])
    x = tf.nn.relu(x)    
    
    # 卷积层4 
    x = tf.nn.conv2d(x, weights['wc4'], strides=[1,m,m,1],padding='SAME')
    x = tf.nn.bias_add(x,biases['bc4'])
    x = tf.nn.relu(x) 
    
    # 卷积层5 
    x = tf.nn.conv2d(x, weights['wc5'], strides=[1,m,m,1],padding='SAME')
    x = tf.nn.bias_add(x,biases['bc5'])
    x = tf.nn.relu(x) 
    
    # 全连接层
    x = tf.reshape(x,[-1,weights['wd1'].get_shape().as_list()[0]])
    x = tf.add(tf.matmul(x,weights['wd1']),biases['bd1'])
    x = tf.nn.relu(x)
    
    # Apply Dropout
    x = tf.nn.dropout(x,dropout)
    # Output, class prediction
    x = tf.add(tf.matmul(x,weights['out']),biases['out'])
    return x

# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 16])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 16, 32])),
    'wc3': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wc4': tf.Variable(tf.random_normal([5, 5, 64, 32])),
    'wc5': tf.Variable(tf.random_normal([5, 5, 32, 16])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([n_steps*n_input*16, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([16])),
    'bc2': tf.Variable(tf.random_normal([32])),
    'bc3': tf.Variable(tf.random_normal([64])),
    'bc4': tf.Variable(tf.random_normal([32])),
    'bc5': tf.Variable(tf.random_normal([16])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


pred = CNN_Net_five(x,weights,biases,dropout=keep_prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred,1),tf.arg_max(y,1))
# tf.argmax(input,axis=None) 由于标签的数据格式是 -1 0 1 3列，该语句是表示返回值最大也就是1的索引，两个索引相同则是预测正确。
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# 更改数据格式，降低均值
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print("**")
    print(training_iters)
    for tr in range(15):
    #for tr in range(3):
        for i in range(int(len(fac)/batch_size)):
            batch_x = fac[i*batch_size:(i+1)*batch_size].reshape([batch_size,n_steps,n_input])
            batch_y = ret[i*batch_size:(i+1)*batch_size].reshape([batch_size,n_classes])
            sess.run(optimizer,feed_dict={x:batch_x,y:batch_y,keep_prob:dropout})
            if(i%50==0):
                print(i,'----',(int(len(fac)/batch_size)))
        loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,y: batch_y, keep_prob:0.8})
        print("Iter " + str(tr*batch_size) + ", Minibatch Loss= " +"{:.26f}".format(loss) + ", Training Accuracy= " +"{:.26f}".format(acc))
    print("Optimization Finished!") 
    print("Accuracy in data set")
    test_data = fac[:batch_size].reshape([batch_size,n_steps,n_input])
    test_label = ret[:batch_size].reshape([batch_size,n_classes])
    loss, acc = sess.run([cost, accuracy], feed_dict={x: test_data,y: test_label, keep_prob:1.})
    print("Accuracy= " +"{:.26f}".format(acc))
    
    print("Accuracy out of data set")
    test_dataT = facT[:len(facT)].reshape([len(facT),n_steps,n_input])
    test_labelT = retT[:len(facT)].reshape([len(facT),n_classes])
    loss, acc = sess.run([cost, accuracy], feed_dict={x: test_dataT,y: test_labelT, keep_prob:1.})
    print("Accuracy= " +"{:.26f}".format(acc))
    
    pred_dataT = predFAC[:batch_size].reshape([1,n_steps,n_input])
    pred_lable = sess.run([pred],feed_dict={x: pred_dataT, keep_prob:1.})
    list_lable = pred_lable[0][0]
    maxindex = np.argmax(list_lable)
    #print("Predict_label is " + str(pred_lable[0][0]))
    if(maxindex==0):
        print("up")
    else:
        print("down")
    sess.close()