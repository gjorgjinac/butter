import os

os.environ["HDF5_DISABLE_VERSION_CHECK"] = "1"
import copy
import os

import numpy as np
import tensorflow as tf
from keras.layers import (LSTM, Bidirectional, Dense, Embedding,
                          SpatialDropout1D, TimeDistributed, concatenate)
from keras.models import Input, Model
from keras.optimizers import Adam, RMSprop
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from model_definition import models
from tf2crf import CRF
from transformers import TFBertMainLayer, TFBertPreTrainedModel
from utils import SentenceGetter, get_embedding_weights, get_label


class FoodNERBertForTokenClassification(TFBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
   
        self.num_labels = config.num_labels
        self.config = config
        self.bert = TFBertMainLayer(self.config, name="bert")
        self.bilstm =  Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))
        self.time_distributed = TimeDistributed(Dense(50, activation="relu"))
        self.crf = CRF(self.num_labels + 1)
        
    def call(self, inputs, **kwargs):
       
        outputs = self.bert(inputs, **kwargs)     
        sequence_output = outputs[0]
        bilstm =self.bilstm(sequence_output)
        td = self.time_distributed(bilstm)
        #return td
        return self.crf(td)   


class BERTCRFModel():
    def get_compiled_model(self, model_to_load, n_tags, full_finetuning=False):

        model = FoodNERBertForTokenClassification.from_pretrained(
          model_to_load, 
          num_labels=n_tags + 1,
          output_attentions = False,
          output_hidden_states = False)
     
        model.summary()
        #optimizer = Adam(learning_rate=3e-5, epsilon=1e-8)
        optimizer = RMSprop(learning_rate=3e-5, epsilon=1e-8)

        model.compile(optimizer=optimizer, loss=model.crf.loss, metrics=[model.crf.accuracy])
        #model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["acc"])
        return model
        

    def process_X(self, data, word2idx, max_sentence_length, tag2idx):
        sentence_getter = SentenceGetter(data, label_adapter=get_label)
        indices = [[word2idx[w[0]] for w in s] for s in sentence_getter.sentences]
        indices = pad_sequences(maxlen=max_sentence_length, sequences=indices, padding="post", value=word2idx["PAD"])
        attention_masks = [[float(i != tag2idx["PAD"]) for i in ii] for ii in indices]
        return indices, attention_masks

    def process_Y(self, data, tag2idx, max_sentence_length, n_tags):
        sentence_getter = SentenceGetter(data, label_adapter=get_label)
        Y = [[tag2idx[w[1]] for w in s] for s in sentence_getter.sentences]
        Y_str = copy.deepcopy(Y)
        Y = pad_sequences(maxlen=max_sentence_length, sequences=Y, padding="post", value=tag2idx["PAD"])
        Y = np.array([to_categorical(i, num_classes=n_tags + 1) for i in Y])
        return Y, Y_str


