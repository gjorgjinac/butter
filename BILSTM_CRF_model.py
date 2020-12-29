import os

os.environ["HDF5_DISABLE_VERSION_CHECK"] = "1"
import copy

import numpy as np
from tensorflow.keras.layers import (LSTM,Input, Bidirectional, Dense, Embedding,
                          SpatialDropout1D, TimeDistributed, concatenate)
from tensorflow.keras.models import  Model
from tensorflow.keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from model_definition import models
from tf2crf import CRF
from utils import SentenceGetter, get_embedding_weights, get_label

class BILSTMCRFModel:

    def get_compiled_model(self, vectorizer_model_name, missing_values_handled,
                           max_sentence_length, max_word_length,
                           n_words, n_tags,
                           word2idx):

        vectorizer_model_settings = models[vectorizer_model_name]
        vectorizer_model_size = vectorizer_model_settings['vector_size']

        word_in = Input(shape=(max_sentence_length,))
        if not vectorizer_model_settings['precomputed_vectors']:
            emb_word = Embedding(input_dim=n_words + 2, output_dim=vectorizer_model_size,
                                 input_length=max_sentence_length,
                                 mask_zero=True)(word_in)
        else:
            embedding_weights = get_embedding_weights(vectorizer_model_name, vectorizer_model_size,
                                                      missing_values_handled,
                                                      word2idx)
            emb_word = Embedding(input_dim=n_words + 2, output_dim=vectorizer_model_size,
                                 input_length=max_sentence_length,
                                 mask_zero=True,
                                 weights=[embedding_weights], trainable=False)(word_in)

        model = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))(emb_word)
        model = TimeDistributed(Dense(50, activation='relu'))(model)
        # print(dir(CRF))
        crf = CRF(n_tags + 1)
        out = crf(model)

        model = Model(word_in, out)
        model.summary()
        model.compile(optimizer="rmsprop", loss=crf.loss, metrics=[crf.accuracy])
        return model

    def process_X(self, data, word2idx, max_sentence_length):
        sentence_getter = SentenceGetter(data, label_adapter=get_label)
        X = [[word2idx[w[0]] for w in s] for s in sentence_getter.sentences]
        X = pad_sequences(maxlen=max_sentence_length, sequences=X, padding="post", value=word2idx["PAD"])
        return X

    def process_Y(self, data, tag2idx, max_sentence_length, n_tags):
        sentence_getter = SentenceGetter(data, label_adapter=get_label)
        Y = [[tag2idx[w[1]] for w in s] for s in sentence_getter.sentences]
        Y_str = copy.deepcopy(Y)
        Y = pad_sequences(maxlen=max_sentence_length, sequences=Y, padding="post", value=tag2idx["PAD"])
        Y = np.array([to_categorical(i, num_classes=n_tags + 1) for i in Y])  # n_tags+1(PAD)
        return Y, Y_str

class BILSTMModel:

    def get_compiled_model(self, vectorizer_model_name, missing_values_handled,
                           max_sentence_length, max_word_length,
                           n_words, n_tags,
                           word2idx):

        vectorizer_model_settings = models[vectorizer_model_name]
        vectorizer_model_size = vectorizer_model_settings['vector_size']

        word_in = Input(shape=(max_sentence_length,))
        if not vectorizer_model_settings['precomputed_vectors']:
            emb_word = Embedding(input_dim=n_words + 2, output_dim=vectorizer_model_size,
                                 input_length=max_sentence_length,
                                 mask_zero=True)(word_in)
        else:
            embedding_weights = get_embedding_weights(vectorizer_model_name, vectorizer_model_size,
                                                      missing_values_handled,
                                                      word2idx)
            emb_word = Embedding(input_dim=n_words + 2, output_dim=vectorizer_model_size,
                                 input_length=max_sentence_length,
                                 mask_zero=True,
                                 weights=[embedding_weights], trainable=False)(word_in)

        model = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))(emb_word)

        out = TimeDistributed(Dense(n_tags + 1, activation="softmax"))(model)

        model = Model(word_in, out)
        model.summary()
        model.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=['acc'])
        return model

    def process_X(self, data, word2idx, max_sentence_length):
        sentence_getter = SentenceGetter(data, label_adapter=get_label)
        X = [[word2idx[w[0]] for w in s] for s in sentence_getter.sentences]
        X = pad_sequences(maxlen=max_sentence_length, sequences=X, padding="post", value=word2idx["PAD"])
        return X

    def process_Y(self, data, tag2idx, max_sentence_length, n_tags):
        sentence_getter = SentenceGetter(data, label_adapter=get_label)
        Y = [[tag2idx[w[1]] for w in s] for s in sentence_getter.sentences]
        Y_str = copy.deepcopy(Y)
        Y = pad_sequences(maxlen=max_sentence_length, sequences=Y, padding="post", value=tag2idx["PAD"])
        Y = np.array([to_categorical(i, num_classes=n_tags + 1) for i in Y])  # n_tags+1(PAD)
        return Y, Y_str
