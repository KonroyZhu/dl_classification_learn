import os
import pickle

import numpy as np
from gensim.models import Word2Vec
from keras import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate

from utils.data_process import load_data, sentence_seg, get_w2v, corpus2sequence


# def get_model(embedding_layer, filter_size=3):
#     model = Sequential()
#     model.add(embedding_layer)
#     model.add(Conv1D(128, filter_size,activation="tanh"))
#     model.add(MaxPooling1D(filter_size))


# return model

def get_embedding(input,word2index, model):
    vocabulary_size = len(word2index)
    output_dim = model.wv.vector_size  # 100

    # prepare embedding matrix 这部分主要是创建一个词向量矩阵，使每个词都有其对应的词向量相对应
    nb_words = len(word2index)
    embedding_matrix = np.zeros((nb_words + 1, output_dim))
    for word, i in word2index.items():
        try:
            embedding_vector = model.wv[word]
            embedding_matrix[i] = embedding_vector
        except:
            pass

    # load pre-trained word embeddings into an Embedding layer
    embedding_layer = Embedding(input_dim=vocabulary_size + 1, output_dim=output_dim, input_length=sequence_length,weights=[embedding_matrix])(
        inputs)  # input_dim !!!记得要等于vocabulary_size+1 !!!weights 用于设置预训练词向量，不设置时随机生成

    # # using a randomly created embedding layer
    # embedding_layer = Embedding(input_dim=vocabulary_size + 1, output_dim=output_dim, input_length=sequence_length)(
    #     inputs)

    return embedding_layer


def get_maxpoop(embedding, filter_size=3):
    reshape = Reshape((sequence_length, embedding_dim, 1))(embedding)
    conv = Conv2D(512, kernel_size=(filter_size, embedding_dim), padding='valid',
                  kernel_initializer='normal', activation='relu')(reshape)
    maxpool = MaxPool2D(pool_size=(sequence_length - filter_size + 1, 1), strides=(1, 1), padding='valid')(conv)

    return maxpool

def get_layers(embedding):
    maxpool_0 = get_maxpoop(embedding_layer, 3)
    maxpool_1 = get_maxpoop(embedding_layer, 4)
    maxpool_2 = get_maxpoop(embedding_layer, 5)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(0.5)(flatten)
    output = Dense(units=2, activation='softmax')(dropout)
    return output


if __name__ == '__main__':


    x_data, y_data = load_data()
    w2v = get_w2v("../model/w2v.pkl")
    corpus = [sentence_seg(x) for x in x_data]
    tokenizer, index2word, word2index = corpus2sequence(corpus, "../model/vocab.pkl")

    sequences = tokenizer.texts_to_sequences(corpus)

    max_sequence_len = max([len(s) for s in sequences])
    print("max sequence lenght:", max_sequence_len)

    data = pad_sequences(sequences, maxlen=max_sequence_len)
    labels = to_categorical(np.asarray(y_data))

    X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=13, test_size=0.3)

    ##########################################################

    sequence_length = max_sequence_len  # 39
    embedding_dim = w2v.wv.vector_size  # 100


    print("Creating Model...")
    inputs = Input(shape=(sequence_length,), dtype='int32')
    embedding_layer = get_embedding(inputs,word2index, w2v)
    print("embedding:",embedding_layer)

    output=get_layers(embedding_layer)

    # this creates a model that includes
    model = Model(inputs=inputs, outputs=output)

    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    print("Traning Model...")
    model.fit(X_train, y_train, batch_size=30  , epochs=100, verbose=1,
              validation_data=(X_test, y_test))  # starts training

    y_pred = model.predict(X_test)
    hit = 0
    for p, l in zip(y_pred, y_test):
        print(p, l)
        if np.argmax(p) == np.argmax(l):
            hit += 1
    print("acc on test:",hit / len(y_test))
