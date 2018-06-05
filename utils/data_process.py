# 停止词
import os
import pickle

from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer



def load_data():
    f1=open("../raw/rt-polarity.neg","r",encoding="utf-8")
    f2=open("../raw/rt-polarity.pos","r",encoding="utf-8")

    neg=f1.readlines()
    pos=f2.readlines()
    f1.close()
    f2.close()

    x_data=pos+neg
    y_data=[1]*len(pos)+[0]*len(neg)

    return x_data,y_data

def sentence_seg(sentence):
    sentence=sentence.replace("\n","")
    stop = stopwords.words('english')
    res=[c for c  in sentence.split(" ") if not c in stop and len(c)>1]
    # print(res)
    return res

def preprocess():
    x_data, y_data = load_data()
    corpus = [sentence_seg(x) for x in x_data]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    sequences = tokenizer.texts_to_sequences(corpus)

    data = pad_sequences(sequences, maxlen=30)

def get_w2v(path):

    if not os.path.exists(path):
        x_data, y_data = load_data()
        corpus = [sentence_seg(x) for x in x_data]
        model = Word2Vec(corpus, size=100, window=5, min_count=5)
        w2v=model
        pickle.dump(model, open(path, "wb"))

    else:
         w2v=pickle.load(open(path,"rb"))
    return w2v

def invert_dict(d):
    return dict((v,k) for k,v in d.items())

def corpus2sequence(corpus,path):
    tokenizer=Tokenizer()
    # tokenizer.fit_on_texts(corpus)
    if not os.path.exists(path):
        #     tokenizer = Tokenizer()
        # tokenizer.fit_on_texts(corpus)
        tokenizer.fit_on_texts(corpus)
        word2index = tokenizer.word_index
        index2word = invert_dict(tokenizer.word_index)

        pickle.dump([tokenizer,index2word,word2index],open(path,"wb"))
    else:
        arr=pickle.load(open(path,"rb"))
        tokenizer=arr[0]
        index2word=arr[1]
        word2index=arr[2]
    return tokenizer,index2word,word2index

