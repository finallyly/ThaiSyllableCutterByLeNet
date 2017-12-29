#-*-coding:utf8-*-
#########################################################################
#   Copyright (C) 2017 All rights reserved.
# 
#   FileName:CNNTry.py
#   Creator: yuliu1finally@gmail.com
#   Time:12/27/2017
#   Description:
#
#   Updates:
#
#########################################################################
#!/usr/bin/python

import os;
import time;
import numpy as np;
import tensorflow as tf;
import TranData2;
BATCH_SIZE=2627;
N_CLASSES=4;
DROUPOUT=0.75;
N_EPOCHES=30;
LEARNING_RATE=0.001;
NUM_SAMPLE=15762;
SKIP_STEP=10;

with tf.name_scope("data"):
    X=tf.placeholder(tf.float32,[None,784],name="X_placeholder");
    Y=tf.placeholder(tf.float32,[None,4],name="Y_placeholder");

dropout = tf.placeholder(tf.float32,name="dropout");
global_step = tf.Variable(0,dtype=tf.int32,trainable=False,name='global_step');

with tf.variable_scope("conv1") as scope:
    images=tf.reshape(X,shape=[-1,28,28,1]);
    kernel=tf.get_variable("kernel",[5,5,1,32],initializer=tf.truncated_normal_initializer());
    #data size 28x28x32
    biases=tf.get_variable("biases",[32],initializer=tf.truncated_normal_initializer());
    conv=tf.nn.conv2d(images,kernel,strides=[1,1,1,1],padding='SAME');
    conv1=tf.nn.relu(conv+biases,name=scope.name);


with tf.variable_scope("pool1") as scope:
    pool1=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME");

#Data size 14x14x32

with tf.variable_scope("conv2") as scope:
    kernel = tf.get_variable("kernel",[5,5,32,64],initializer=tf.truncated_normal_initializer());
    biases = tf.get_variable("biases",[64],initializer=tf.truncated_normal_initializer());
    conv = tf.nn.conv2d(pool1,kernel,strides=[1,1,1,1],padding="SAME");
    conv2=tf.nn.relu(conv+biases,name=scope.name);
#Data size 14x14x64

with tf.variable_scope("pool2") as scope:
    pool2=tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME");
#Data size 7x7x64


with tf.variable_scope("fc") as scope:
    input_feature_size=7*7*64;
    w=tf.get_variable("weights",[input_feature_size,1024],initializer=tf.truncated_normal_initializer());
    b=tf.get_variable("biases",[1024],initializer=tf.truncated_normal_initializer());
    pool2 = tf.reshape(pool2,[-1,input_feature_size]);
    fc=tf.nn.relu(tf.matmul(pool2,w)+b,name="relu");
    fc = tf.nn.dropout(fc,DROUPOUT,name="relu_dropout");

with tf.variable_scope("softmax") as scope:
    w=tf.get_variable("weights",[1024,N_CLASSES],initializer=tf.truncated_normal_initializer());
    b=tf.get_variable("biases",[N_CLASSES],initializer=tf.truncated_normal_initializer());
    logits=tf.matmul(fc,w)+b;

with tf.name_scope("loss"):
    entropy=tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=logits);
    loss = tf.reduce_mean(entropy,name="loss");


with tf.name_scope("summaries"):
    tf.summary.scalar("loss",loss);
    tf.summary.histogram("histgram_loss",loss);
    summary_op = tf.summary.merge_all();

optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss,global_step=global_step);


source_train, target_train = TranData2.LoadTrainData("train.vector.20171207.txt");
source_test, target_test,labels = TranData2.LoadTestData("test.vector.txt");
'''
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer());
    saver =tf.train.Saver();
    writer = tf.summary.FileWriter("./graphs/convnet",sess.graph);
    ckpt=tf.train.get_checkpoint_state("checkpoints/convnet_mnist/checkpoint");
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess,ckpt.model_checkpoint_path);


    initial_step = global_step.eval();

    start_time = time.time();
    n_batches = NUM_SAMPLE/BATCH_SIZE;
    total_loss = 0.0;

    for index in range(0,N_EPOCHES):
        for X_Batch,Y_Batch in  TranData2.batch_data(source_train,target_train,BATCH_SIZE):
            _,loss_batch,summary = sess.run([optimizer,loss,summary_op],feed_dict={X:X_Batch,Y:Y_Batch,dropout:DROUPOUT});
            writer.add_summary(summary,global_step=index);
            total_loss+=loss_batch;
            print('Average loss at step {}: {:5.1f}'.format(index + 1, total_loss / SKIP_STEP))
            total_loss = 0.0
            saver.save(sess, 'checkpoints/convnet_mnist/mnist-convnet', index);
    print("Optimization Finished!");
    print("Total time: {0} seconds".format(time.time() - start_time));

'''

with  tf.Session() as sess:

    #Predict
    checkpoint = tf.train.latest_checkpoint('checkpoints/convnet_mnist');
    saver=tf.train.Saver();
    saver.restore(sess, checkpoint);
    fout = open("predict.txt","w");
    for i in range(0,len(source_test)):
        x_sentence = source_test[i];
        y_sentence = target_test[i];
        #print x_sentence;
        #print y_sentence;
        labels_sentence = labels[i];
        _, loss_batch, logits_batch=sess.run([optimizer,loss,logits],feed_dict={X:x_sentence,Y:y_sentence,dropout:1.0});
        preds = tf.nn.softmax(logits);
        predict_labels = tf.argmax(preds, 1);
        predict_labels_array=sess.run(predict_labels,feed_dict={X:x_sentence,Y:y_sentence,dropout:1.0});
        for j in  range(0,len(source_test[i])):
            fout.write("%s\t%s\n"%(labels_sentence[j],TranData2.GetLabelById(predict_labels_array[j])))
            #print predict_labels_array[j];
        fout.write("\n");
    fout.close();


















