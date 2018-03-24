#encoding=utf-8
from gensim.models import word2vec
import numpy as np
import  tensorflow as tf

def getTrainData(n,batchSize,seq_lenth=200,emb_lenth=200):
    fread1=open("../Resource/Data/fenci.txt",'r',encoding='utf-8')
    fread2 = open("../Resource/Data/labels.txt", 'r', encoding='utf-8')
    lines=fread1.readlines()
    #——————————————————————————————————后期改进——————————————————————————————————————————————
    text=lines[n*batchSize+5900:5900+(n+1)*batchSize]
    labels=np.array(fread2.readlines()[n*batchSize+5900:5900+(n+1)*batchSize])
    return getWordEmbedding(text,batchSize,seq_lenth,emb_lenth),labels

def getTestData(seq_lenth=200,emb_lenth=200):
    fread1 = open("../Resource/Data/fenci.txt", 'r', encoding='utf-8')
    fread2 = open("../Resource/Data/labels.txt", 'r', encoding='utf-8')
    lines = fread1.readlines()
    text = lines[12000:16000]
    labels = fread2.readlines()[12000:16000]
    return getWordEmbedding(text, 4000, seq_lenth, emb_lenth), labels


def wordEmbedding():
    sentences = word2vec.Text8Corpus("../Resource/Data/fenci.txt",)
    model = word2vec.Word2Vec(sentences,size=200, min_count=1)
    model.save("../Resource/Data/wordembedding.bin")
    model.init_sims(replace=True)

def getWordEmbedding(text,batchSize,sep_lenth=200,emb_size=200):
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
    pass
    # wordEmbedding()
    # print(getTrainData(0,2)[0])