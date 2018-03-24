#encoging=utf-8
import pandas as pd
import numpy as np
import jieba
def loadfile():
    neg=pd.read_excel('../Resource/Data/neg.xls',header=None,index=None)
    pos=pd.read_excel('../Resource/Data/pos.xls',header=None,index=None)

    combined=np.concatenate((pos[0], neg[0]))
    y = np.concatenate((np.ones(len(pos),dtype=int), np.zeros(len(neg),dtype=int)))

    return combined,y

#对句子进行分词，并去掉换行符
def tokenizer(text,labels):
    ''' Simple Parser converting each document to lower-case, then
        removing the breaks for new lines and finally splitting on the
        whitespace
    '''
    f1 = open("../Resource/Data/fenci.txt", 'a', encoding='utf-8')
    f2 = open("../Resource/Data/labels.txt", 'a', encoding='utf-8')
    i=0
    for line in text:
        f2.write(str(labels[i]))
        f2.write('\n')
        i += 1
        seg_list = jieba.cut(line,cut_all=False)
        f1.write(" ".join(seg_list).replace(',','').replace('，','').replace('.','').replace('。','').replace('、','').replace('!','').replace('！','').replace('(','')
                     .replace(')','').replace('（','').replace('）','').replace('[','').replace(']','').replace('?','').replace('？','').replace('“','').replace('”','')
                     .replace('：','').replace(':','').replace('-','').replace('~','').replace('…','').replace("【",'').replace('】','').replace('《','').replace('》',''))
        f1.write('\n')
    # return text
    f1.close()
    f2.close()

def lodadictionary():
    f2=open("../Resource/Data/fenci.txt",'r',encoding='utf-8')
    lines=f2.readlines()
    print(lines[0])

if __name__ == '__main__':
    pass
    sourceData,y=loadfile()
    tokenizer(sourceData,y)
    # lodadictionary()