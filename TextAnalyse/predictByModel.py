#encoding=utf-8
import tensorflow as tf

#利用已训练好的模型来预测
def predict(x_data,embedding_size):
    modelpath='../Resource/Data/CNN-LSTM_Model'
    saver = tf.train.Saver()
    y=tf.placeholder(dtype=tf.float32,shape=[1],name='predict_y')
    x=tf.placeholder(dtype=tf.float32,shape=[200,embedding_size],name='input_x')
    with tf.Session as sess:
        saver.restore(sess, modelpath)
        result = sess.run(y, feed_dict={x: x_data})
        if(result==1):
            print("此文本的情感为积极。")
        else:
            print("此文本的情感为消极。")