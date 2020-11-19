import re
import pandas as pd
from nltk import word_tokenize
import numpy as np
import gensim.downloader as api
from num2words import num2words

from model_definition import models


def handle_missing_value(lowercase_token, model, vector_size):
    try:
        lowercase_token = num2words(lowercase_token)
    except Exception:
        pass
    token_without_punctuation = re.sub(r"[,.;@#-?!&$]+\ *", " ", lowercase_token)
    words_in_token_and_vocab = set(word_tokenize(token_without_punctuation)).intersection(set(model.vocab))
    words_in_token_and_vocab = [model[word] for word in words_in_token_and_vocab]
    if len(words_in_token_and_vocab) > 0:
        return np.mean(words_in_token_and_vocab, axis=0) if len(words_in_token_and_vocab) > 1 else \
            words_in_token_and_vocab[0]
    else:
        return np.zeros(vector_size)


def get_token_vector(token, model, vector_size, handle_missing_values=False):
    lowercase_token = token.lower()

    if lowercase_token in model.vocab:
        return model[lowercase_token]
    else:
        if handle_missing_values:
            return handle_missing_value(lowercase_token, model, vector_size)
        return np.zeros(vector_size)


df = pd.read_table('data/full-food-classification.txt', delimiter='\t', header=0)

handle_missing_values = False
for model_name in models.keys():
    m = models[model_name]
    model = api.load(model_name)
    vector_dict = {}
    for token in set(df.index):
        vector_dict[token.lower()] = get_token_vector(token, model, m['vector_size'], handle_missing_values)

    file_name = f'vectors/missing_values_handled_{handle_missing_values}/{model_name}'
    pd.DataFrame.from_dict(vector_dict, orient='index').to_csv(file_name)
