import numpy as np
from keras import Sequential, Input, Model
from keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from utils.data_process import load_data, sentence_seg, corpus2sequence


def get_layer(embedding):
    # building model
    biLSTM = Bidirectional(LSTM(128, implementation=2))(embedding)
    dropout = Dropout(0.5)(biLSTM)
    dense = Dense(2, activation="relu")(dropout)

    dropout = Dropout(0.5)(dense)
    output = Dense(units=2, activation='softmax')(dropout)

    model = Model(inputs=inputs, outputs=output)  # from tensor to model

    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def get_layer2(embedding):
    return 0


if __name__ == '__main__':
    x_data, y_data = load_data()
    corpus = [sentence_seg(x) for x in x_data]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)

    sequences = tokenizer.texts_to_sequences(corpus)

    max_sequence_len = max([len(s) for s in sequences])
    print("max sequence lenght:", max_sequence_len)

    data = pad_sequences(sequences, maxlen=max_sequence_len)
    labels = to_categorical(np.asarray(y_data))

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)

    ##########################################################

    inputs = Input(shape=(max_sequence_len,), dtype='int32')
    embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_sequence_len)(
        inputs)
    print("embedding:", embedding_layer)

    # model = get_layer(embedding_layer)
    model=get_layer2(embedding_layer)

    print("Traning Model...")
    model.fit(X_train, y_train, batch_size=30, epochs=100, verbose=1,
              # validation_data=(X_train[:len(y_test)], y_test))  # starts training
              validation_data=(X_test, y_test))  # starts training

    y_pred = model.predict(X_test)
    hit = 0
    for p, l in zip(y_pred, y_test):
        print(p, l)
        if np.argmax(p) == np.argmax(l):
            hit += 1
    print("acc on test:", hit / len(y_test))
