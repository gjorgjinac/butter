import os

os.environ["HDF5_DISABLE_VERSION_CHECK"] = "1"

from tf2crf import CRF
from keras_preprocessing.sequence import pad_sequences
import numpy as np
from model_definition import models
import copy
from utils import get_embedding_weights, SentenceGetter, get_label
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Input, Bidirectional, concatenate, SpatialDropout1D
from tensorflow.keras.utils import to_categorical

class BILSTMCRFDoubleInputModel:

    def get_compiled_model(self, vectorizer_model_name, missing_values_handled,
                           max_sentence_length, max_word_length,
                           n_words, n_chars, n_tags,
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

        # input and embeddings for characters
        char_in = Input(shape=(max_sentence_length, max_word_length,))
        emb_char = TimeDistributed(Embedding(input_dim=n_chars + 2, output_dim=10, mask_zero=True))(char_in)
        # character LSTM to get word encodings by characters
        char_enc = TimeDistributed(LSTM(units=20, return_sequences=False,
                                        recurrent_dropout=0.5))(emb_char)

        # main LSTM
        x = concatenate([emb_word, char_enc])
        # x = SpatialDropout1D(0.3)(x)

        model = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))(x)
        model = TimeDistributed(Dense(50, activation='relu'))(model)

        crf = CRF(n_tags + 1)
        out = crf(model)

        model = Model([word_in, char_in], out)
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



class BILSTMDoubleInputModel:

    def get_compiled_model(self, vectorizer_model_name, missing_values_handled,
                           max_sentence_length, max_word_length,
                           n_words, n_chars, n_tags,
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

        # input and embeddings for characters
        char_in = Input(shape=(max_sentence_length, max_word_length,))
        emb_char = TimeDistributed(Embedding(input_dim=n_chars + 2, output_dim=10, mask_zero=True))(char_in)
        # character LSTM to get word encodings by characters
        char_enc = TimeDistributed(LSTM(units=20, return_sequences=False,
                                        recurrent_dropout=0.5))(emb_char)

        # main LSTM
        x = concatenate([emb_word, char_enc])
        x = SpatialDropout1D(0.3)(x)
        main_lstm = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.6))(x)
        out = TimeDistributed(Dense(n_tags + 1, activation="softmax"))(main_lstm)

        model = Model([word_in, char_in], out)
        model.summary()
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
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
