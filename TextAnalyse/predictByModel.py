#encoding=utf-8
import tensorflow as tf
import TextAnalyse.provideData as pd
import numpy as np
#利用已训练好的模型来预测
def predict(x_data,embedding_size):
    modelpath='../Resource/Data/CNN-LSTM_Model'
    saver = tf.train.Saver()
    y=tf.placeholder(dtype=tf.float32,shape=[1],name='predict_y')
    x=tf.placeholder(dtype=tf.float32,shape=[200,embedding_size],name='input_x')
    with tf.Session() as sess:
        saver.restore(sess, modelpath)
        result = sess.run(y, feed_dict={x: x_data})
        if(result==1):
            print("此文本的情感为积极。")
        else:
            print("此文本的情感为消极。")

def testModel(sampleSize,sequence_length, num_classes, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0,num_hidden=100,epoch=100,learning_rate=0.1,batchSize=1,point=0.5,keep_prob=0.5):
    modelpath='../Resource/Data/CNN-LSTM_Model.ckpt'
    with tf.Graph().as_default() as g:
        x=tf.placeholder(dtype=tf.float32,shape=[batchSize,sequence_length,embedding_size,1],name='input_x')
        # tf.summary.histogram('input/input_x',x)
        y = tf.placeholder(dtype=tf.float32, shape=[batchSize,1],name='labels')
        dropout_keep_prob=tf.placeholder(dtype=tf.float32,name='keep_prob')
        modelpath='../Resource/Data/CNN-LSTM_Model.ckpt'
        # pointNum = tf.constant(point,tf.float32,[1],'ponit-num')
        #定义卷积层，由于需要保持语句的时序信息，故此处不需要池化层
        conv_w=tf.get_variable('convweights',[filter_sizes,embedding_size,1,num_filters],initializer=tf.truncated_normal_initializer(stddev=0.1))
        tf.add_to_collection('l2-reg-conv',conv_w)
        conv_bias=tf.get_variable('convbias',[num_filters],dtype=tf.float32,initializer=tf.zeros_initializer())
        conv1=tf.nn.bias_add(tf.nn.conv2d(x,conv_w,[1,1,1,1],padding='VALID'),conv_bias,name='conv1')
        active_conv1=tf.nn.tanh(conv1,'conv-relu')
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
        lstm = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, state_is_tuple=True)
        lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=dropout_keep_prob)
        initial_state = lstm.zero_state(batchSize, tf.float32)
        lstm_in=tf.transpose(active_conv1,[2,1,0,3])
        lstm_in=tf.reshape(lstm_in,[-1,num_filters])
        lstm_in=tf.split(lstm_in,sequence_length-filter_sizes+1,0)
        outputs, final_state = tf.contrib.rnn.static_rnn(lstm, lstm_in, dtype=tf.float32,initial_state = initial_state)
        result = tf.layers.dense(outputs[-1], 1, name='dense')
        # tf.summary.scalar('Dense/dense',result)
        predictions = tf.nn.sigmoid(result,'predictions')
        with tf.Session() as sess:
            # saver = tf.train.import_meta_graph('../Resource/Data/CNN-LSTM_Model.meta')
            saver = tf.train.Saver()
            saver.restore(sess, modelpath)
            # graph = tf.get_default_graph()
            # preditions = graph.get_tensor_by_name("preditions")
            # logits = graph.get_tensor_by_name("dense:0")
            right_num = 0
            right_flag = tf.reduce_sum(tf.cast(tf.equal(y,tf.round(predictions)),tf.float32))
            x_test,y_test = pd.getTestData(0,sampleSize,sequence_length,embedding_size)
            x_test = np.reshape(x_test,(sampleSize,1,sequence_length,embedding_size,1))
            y_test = np.asarray(y_test,np.float32)
            for i in range(sampleSize):
                x_data = x_test[i]
                y_data = y_test[i]
                y_data = np .reshape(y_data,(1,1))
                right,result = sess.run([right_flag,predictions],feed_dict={x: x_data,y:y_data,dropout_keep_prob:1})
                right_num += right
            acc = right_num/sampleSize
            print(acc)



if __name__ == '__main__':
    testModel(sampleSize=4000,sequence_length=100,num_classes=2,embedding_size=100,filter_sizes=10,num_filters=10,batchSize=1,keep_prob=1)

