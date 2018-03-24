#encoding=utf-8
import tensorflow as tf
import numpy as np
import TextAnalyse.provideData as pd

def createNetwork(sequence_length, num_classes, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0,num_hidden=100,epoch=100,learning_rate=0.1,batchSize=50,point=0.5,keep_prob=0.5):
    x=tf.placeholder(dtype=tf.float32,shape=[None,sequence_length,embedding_size,1],name='input_x')
    y = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')
    dropout_keep_prob=tf.placeholder(dtype=tf.float32,name='keep_prob')
    modelpath='../Resource/Data/CNN-LSTM_Model'
    # pointNum = tf.constant(point,tf.float32,[1],'ponit-num')
    #定义卷积层，由于需要保持语句的时序信息，故此处不需要池化层
    conv_w=tf.get_variable('convweights',[filter_sizes,embedding_size,1,num_filters],initializer=tf.truncated_normal_initializer(stddev=0.1))
    tf.add_to_collection('l2-reg-conv',conv_w)
    conv_bias=tf.get_variable('convbias',[num_filters],dtype=tf.float32,initializer=tf.zeros_initializer())
    conv1=tf.nn.bias_add(tf.nn.conv2d(x,conv_w,[1,1,1,1],padding='VALID'),conv_bias,name='conv1')
    active_conv1=tf.nn.relu(conv1)
    conv1_drop=tf.nn.dropout(active_conv1,dropout_keep_prob,name='dropout')

    #处理卷积层的输出，从而作为LSTM每一时间步的输入
    out_weight = tf.Variable(tf.random_normal([num_hidden, 1]))
    tf.add_to_collection('l2-reg-conv-lstm', out_weight)
    out_bias = tf.Variable(tf.random_normal([1]))
    # dropout_keep_prob = keep_prob
    # predictions = tf.get_variable('predict',shape=[1],dtype=tf.float32,initializer=tf.zeros_initializer())
    predictions = 0
    # tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # 定义LSTM单元
    lstm = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, state_is_tuple=True)
    lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=dropout_keep_prob)
    state = lstm.zero_state(num_filters, tf.float32)
    with tf.Session() as sess:
        for k in range(epoch):
            epochcost = 0
            #----------------------------------------------------------------------------后期修改训练集---------------------------------------------------------------
            for i in range(int(2/batchSize)):
                x_data,y_data=pd.getTrainData(i,batchSize,sequence_length,embedding_size)
                x_data = x_data.reshape((batchSize,sequence_length,embedding_size,1))
                # sess.run(tf.global_variables_initializer())
                # conv_seq=sess.run([conv1_drop],feed_dict={x:x_data,y:y_data,dropout_keep_prob:0.5})
                # conv_seq=np.array(conv1_drop)
                for j in range(batchSize):
                    for i in range(sequence_length-filter_sizes+1):
                        current_input = conv1_drop[j,i,:,:]
                        current_input = tf.reshape(current_input,(num_filters,1))
                        # current_input = tf.convert_to_tensor(current_input,dtype=tf.float32,name='lstm-input')
                        #定义LSTM层
                        output,state=lstm(current_input, state)
                finaloutput=output
                fullconnect = tf.nn.bias_add(tf.matmul(finaloutput,out_weight),out_bias)
                # fullconnect = tf.nn.xw_plus_b(finaloutput, out_weight, out_bias, name='fullconnect')
                predictions = tf.nn.sigmoid(tf.reduce_mean(fullconnect), name="predictions")
                # predictions_num = np.array(predictions)
                # if predictions.eval()<point:
                #     predictions=0
                # else:
                #     predictions=1
                loss = tf.square(y[j]-predictions)
                regularization_cost=l2_reg_lambda/batchSize*(tf.reduce_sum(tf.nn.l2_loss(tf.get_collection('l2-reg-conv-lstm')))+tf.reduce_sum(tf.nn.l2_loss(tf.get_collection('l2-reg-conv')))+tf.reduce_sum(tf.nn.l2_loss(tf.get_collection(tf.GraphKeys.WEIGHTS))))
                cost = loss + regularization_cost
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate,).minimize(cost)
                if k == 0:
                    sess.run(tf.global_variables_initializer())
                cost,_ = sess.run([cost,optimizer],feed_dict={x:x_data,y:y_data,dropout_keep_prob:keep_prob})
                epochcost += cost
            print('周期', k + 1, ':', epochcost)
        saver = tf.train.Saver()
        saver.save(sess,modelpath)
        accurancy = tf.reduce_mean(tf.cast(tf.equal(y, tf.round(predictions)), tf.float32))
        x_test,y_test= pd.getTestData()
        acc = sess.run(accurancy, feed_dict={x: x_test, y: y_test,dropout_keep_prob:1})
        print(acc)

if __name__ == '__main__':
    createNetwork(sequence_length=100,num_classes=2,embedding_size=200,filter_sizes=5,num_filters=10,l2_reg_lambda=0.1,epoch=10,learning_rate=0.5,batchSize=2)