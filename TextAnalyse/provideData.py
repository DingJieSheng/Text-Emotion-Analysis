#encoding=utf-8
from gensim.models import word2vec
import numpy as np
from sklearn.model_selection import train_test_split as tts

def getTrainData(n,sampleSize,seq_lenth=200,emb_lenth=100):
    fread1=open("../Resource/Data/fenci.txt",'r',encoding='utf-8')
    fread2 = open("../Resource/Data/labels.txt", 'r', encoding='utf-8')
    lines=fread1.readlines()
    #——————————————————————————————————后期改进——————————————————————————————————————————————
    text=lines[0:12000]
    # text=lines[n*batchSize:(n+1)*batchSize]
    labels=np.array(fread2.readlines()[0:12000])
    # labels=np.array(fread2.readlines()[n*batchSize:(n+1)*batchSize])
    text,text_test,labels,labels_test = tts(text,labels,test_size=1-sampleSize/12000)
    labels = labels.reshape((sampleSize,1))
    return getWordEmbedding(text,sampleSize,seq_lenth,emb_lenth),labels

def getTestData(n,batchSize,seq_lenth=200,emb_lenth=100):
    fread1 = open("../Resource/Data/fenci.txt", 'r', encoding='utf-8')
    fread2 = open("../Resource/Data/labels.txt", 'r', encoding='utf-8')
    lines = fread1.readlines()
    text = lines[12000+n*batchSize:12000+(n+1)*batchSize]
    labels =np.array(fread2.readlines()[12000+n*batchSize:12000+(n+1)*batchSize])
    labels = labels.reshape((batchSize,1))
    return getWordEmbedding(text, batchSize, seq_lenth, emb_lenth), labels


def wordEmbedding():
    sentences = word2vec.Text8Corpus("../Resource/Data/fenci.txt",)
    model = word2vec.Word2Vec(sentences,size=100, min_count=5)
    model.save("../Resource/Data/wordembedding.bin")
    model.init_sims(replace=True)

def getWordEmbedding(text,batchSize,sep_lenth=200,emb_size=100):
    words_embedding=[]
    model = word2vec.Word2Vec.load("../Resource/Data/wordembedding.bin")
    for line in text:
        for i in range(sep_lenth):#控制句长为200
            try:
                word=line.split(' ')[i]
            except:
                word='.'#如果句子不够长就用不存在与字典中的符号代替，从而使得句子长度一致
            try:
                word_embedding=model[word]
            except:
                word_embedding=[0]*emb_size#不在字典中的字词用全零向量表示
            words_embedding.append(word_embedding)
    words_embedding = np.array(words_embedding)
    words_embedding = words_embedding.reshape((batchSize,sep_lenth,emb_size,1))
    return words_embedding



if __name__ == '__main__':
    # getTrainData(0,16000,200,100)
    # wordEmbedding()
    getTrainData(0,400,200,100)