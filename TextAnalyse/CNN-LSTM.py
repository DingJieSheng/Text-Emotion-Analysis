#encoding=utf-8
import tensorflow as tf
import numpy as np
import TextAnalyse.provideData as pd

def createNetwork(sampleSize,sequence_length, num_classes, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.1,num_hidden=100,epoch=100,learning_rate=0.1,batchSize=50,point=0.5,keep_prob=0.5):
    with tf.name_scope('input'):
        x=tf.placeholder(dtype=tf.float32,shape=[batchSize,sequence_length,embedding_size,1],name='input_x')
        # tf.summary.histogram('input/input_x',x)
        y = tf.placeholder(dtype=tf.float32, shape=[batchSize,1],name='labels')
        # tf.summary.histogram('input/labels',y)
        dropout_keep_prob=tf.placeholder(dtype=tf.float32,name='keep_prob')
        # tf.summary.histogram('input/keep_prob',dropout_keep_prob)
        modelpath='../Resource/Data/CNN-LSTM-Models/CNN-LSTM_Model-3.ckpt'
        # pointNum = tf.constant(point,tf.float32,[1],'ponit-num')
    #定义卷积层，由于需要保持语句的时序信息，故此处不需要池化层
    with tf.name_scope('conv'):
        conv_w=tf.get_variable('convweights',[filter_sizes,embedding_size,1,num_filters],initializer=tf.truncated_normal_initializer(stddev=0.1))
        # tf.summary.histogram('conv/convweights',conv_w)
        tf.add_to_collection('l2-reg-conv',conv_w)
        conv_bias=tf.get_variable('convbias',[num_filters],dtype=tf.float32,initializer=tf.zeros_initializer())
        # tf.summary.histogram('conv/convbias',conv_bias)
        conv1=tf.nn.bias_add(tf.nn.conv2d(x,conv_w,[1,1,1,1],padding='VALID'),conv_bias,name='conv1')
        # tf.summary.scalar('conv/conv1',conv1)
        active_conv1=tf.nn.tanh(conv1,'conv-relu')
        # tf.summary.scalar('conv/conv-relu',active_conv1)
    # conv1_drop=tf.nn.dropout(active_conv1,dropout_keep_prob,name='dropout')

    #处理卷积层的输出，从而作为LSTM每一时间步的输入
    # out_weight = tf.Variable(tf.random_normal([num_hidden, 1]))
    # tf.add_to_collection('l2-reg-conv-lstm', out_weight)
    # out_bias = tf.Variable(tf.random_normal([1]))
    # dropout_keep_prob = keep_prob
    # predictions = tf.get_variable('predict',shape=[1],dtype=tf.float32,initializer=tf.zeros_initializer())
    predictions = 0
    # tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # 定义LSTM单元
    with tf.name_scope('LSTM'):
        lstm = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, state_is_tuple=True)
        lstm = tf.nn.rnn_cell.DropoutWrapper(lstm,output_keep_prob=dropout_keep_prob)
        initial_state = lstm.zero_state(batchSize, tf.float32)
        lstm_in=tf.transpose(active_conv1,[2,1,0,3],'conv-into-lstm')
        # tf.summary.scalar('LSTM/conv-into-lstm',lstm_in)
        lstm_in=tf.reshape(lstm_in,[-1,num_filters])
        lstm_in=tf.split(lstm_in,sequence_length-filter_sizes+1,0,name='lstm-split')
        # tf.summary.scalar('LSTM/lstm-split',lstm_in)
        outputs, final_state = tf.contrib.rnn.static_rnn(lstm, lstm_in, dtype=tf.float32,initial_state = initial_state)

    with tf.name_scope('Dense'):
        result = tf.layers.dense(outputs[-1], 1, name='dense')
        # tf.summary.scalar('Dense/dense',result)
        predictions = tf.nn.sigmoid(result,'predictions')
        # tf.summary.scalar('Dense/predictions',predictions)
    # loss = tf.reduce_sum(tf.square(predictions-y))
    with tf.name_scope('costFunc'):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=result,name='loss')
        # tf.summary.scalar('costFunc/loss',loss)
        #------------------------------------------------------------------前期由于训练集小，后期需加入正则化项---------------------------------------------
        regularization_cost = l2_reg_lambda * (tf.reduce_sum(tf.nn.l2_loss(tf.get_collection('l2-reg-conv')))
                                               + tf.reduce_sum(tf.nn.l2_loss(tf.get_collection(tf.GraphKeys.WEIGHTS))))
        costFunc = tf.reduce_sum(loss)+ regularization_cost
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(costFunc)
        # tf.summary.scalar('costFunc/optimizer',optimizer)
    with tf.name_scope('trainAndTestNet'):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            # merged = tf.summary.merge_all() #将图形、训练过程等数据合并在一起
            # writer = tf.summary.FileWriter('../Resource/Data/logs',sess.graph) #将训练日志写入到logs文件夹下
            # writer = tf.summary.FileWriter("../Resource/Data/logs/", sess.graph)
            # sess.run(tf.global_variables_initializer())
            x_data,y_data=pd.getTrainData(-1,sampleSize,sequence_length,embedding_size)
            for k in range(epoch):
                epochcost = 0
                #----------------------------------------------------------------------------后期修改训练集---------------------------------------------------------------
                for i in range(int(sampleSize/batchSize)):
                    x_train,y_train=x_data[i*batchSize:(i+1)*batchSize],y_data[i*batchSize:(i+1)*batchSize]
                    # x_data = x_data.reshape((batchSize,sequence_length,embedding_size,1))
                    # y_data = y_data.reshape((batchSize,1))
                    # sess.run(tf.global_variables_initializer())
                    # conv_seq=sess.run([conv1_drop],feed_dict={x:x_data,y:y_data,dropout_keep_prob:0.5})
                    # conv_seq=np.array(conv1_drop)
                    cost,_ ,pre,res= sess.run([costFunc,optimizer,predictions,result],feed_dict={x:x_train,y:y_train,dropout_keep_prob:keep_prob})
                # if k%5==0:
                #     writer.add_summary(merged,k) #将日志数据写入文件
                    epochcost += cost
                print('周期', k + 1, ':', epochcost)
            saver.save(sess,modelpath)
            accurancy = tf.reduce_sum(tf.cast(tf.equal(y, tf.round(predictions)), tf.float32))
            acc_num = 0
            # x_test,y_test= pd.getTestData(seq_lenth=sequence_length,emb_lenth=embedding_size)
            for i in range(int(4000/batchSize)):
                x_test,y_test= pd.getTestData(n=i,batchSize=batchSize,seq_lenth=sequence_length,emb_lenth=embedding_size)
                # x_test,y_test= pd.getTrainData(n=i,batchSize=batchSize,seq_lenth=sequence_length,emb_lenth=embedding_size)
                acc,result = sess.run([accurancy,predictions], feed_dict={x: x_test, y: y_test,dropout_keep_prob:1})
                acc_num += acc
            print(acc_num/4000)

if __name__ == '__main__':
    createNetwork(sampleSize=12000,sequence_length=100,num_classes=2,embedding_size=100,filter_sizes=10,num_filters=10,l2_reg_lambda=1,epoch=100,learning_rate=0.001,batchSize=100,keep_prob=0.5)