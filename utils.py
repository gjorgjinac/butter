import os
import re

import pandas as pd
import numpy as np
import pickle


class SentenceGetter(object):

    def __init__(self, data, label_adapter):
        self.n_sent = 0
        self.data = data
        self.empty = False

        self.grouped = []
        sentence = []
        for key, value in zip(data.iloc[:, 0].keys(), data.iloc[:, 0].values):
            sentence.append((key, label_adapter(value)))
            # print(label_adapter(value))
            if key == '.':
                self.grouped.append(sentence)
                sentence = []
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.sentences[self.n_sent]
            self.n_sent += 1
            return s
        except:
            return None


def aggregate_report_pkl(report, vectorizer_model_name, file_name, n_epochs, nn_model_name='BILSTM_CharEmb'):
    base = os.path.join("results", nn_model_name, vectorizer_model_name)
    os.makedirs(base, exist_ok=True)

    if os.path.exists(os.path.join(base, file_name)):
        with open(os.path.join(base, file_name), "rb") as f_in:
            rez = pickle.load(f_in)
    else:
        rez = []

    rez.append(report)
    with open(os.path.join(base, file_name), "wb") as f_out:
        pickle.dump(rez, f_out)


def save_report_to_file(report, vectorizer_model_name, file_name, n_epochs, which_fold=None,
                        nn_model_name='BILSTM_CharEmb'):
    base = os.path.join("results", nn_model_name, vectorizer_model_name)
    os.makedirs(base, exist_ok=True)

    ret = pd.DataFrame.from_dict(report)
    ltx = ret.to_latex(label="tab:results", caption="Results")
    print(ret)
    with open(os.path.join(base, file_name), "a") as f_out:
        f_out.write(ltx+"\n")
        f_out.write("Final Epochs = {n_epochs}\n".format(n_epochs=n_epochs))
        if which_fold:
            f_out.write("Fold = {which_fold}\n".format(which_fold=which_fold))


def get_pred_and_ground_string(Y_test, predictions, idx2tag):
    preds_str = []
    ground_str = []

    print(Y_test.shape)
    print(predictions.shape)

    len_dict = dict()
    for idx1, el in enumerate(Y_test):
        len_dict[idx1] = int(1e5)
        for idx2, el2 in enumerate(el):
            j = idx2tag[np.argmax(el2) if len(Y_test.shape) >= 3 else el2]
            if j == 'PAD':
                len_dict[idx1] = idx2
                break
            ground_str.append(j)

    for idx1, el in enumerate(predictions):
        for idx2, el2 in enumerate(el):
            j = idx2tag[np.argmax(el2) if len(predictions.shape) >= 3 else el2]
            if idx2 >= len_dict[idx1]:
                break
            preds_str.append(j)

    return preds_str, ground_str


def transform_label(l):
    # print(re.search(r'^([BI]).*', l))
    return re.sub(r'^([BI]).*', r'\1-FOOD', l)


def get_label(l):
    return l


def get_char_indices(data, max_len_char, max_sentence_length, char2idx):
    sentence_getter = SentenceGetter(data, label_adapter=get_label)
    X_char = []
    for sentence in sentence_getter.sentences:
        sent_seq = []
        for i in range(max_sentence_length):
            word_seq = []
            for j in range(max_len_char):
                try:
                    word_seq.append(char2idx.get(sentence[i][0][j]))
                except:
                    word_seq.append(char2idx.get("PAD"))
            sent_seq.append(word_seq)
        X_char.append(np.array(sent_seq))
    return X_char


def get_embedding_weights(vectorizer_model_name, vectorizer_model_size, missing_values_handled, word2idx):
    bpath = "vectors"
    vectors_path = os.path.join(bpath,
                                'missing_values_handled_{missing_values_handled}/{vectorizer_model_name}'.format(
                                    missing_values_handled=missing_values_handled,
                                    vectorizer_model_name=vectorizer_model_name))
    vectors = pd.read_csv(vectors_path, index_col=[0])
    embedding_weights = [vectors.loc[word.lower(), :] if word in vectors.index else np.zeros(vectorizer_model_size) for
                         word, index in word2idx.items()]
    embedding_weights = np.array(embedding_weights)
    print(embedding_weights.shape)
    return embedding_weights


def get_char_to_index_dict(words):
    chars = set([w_i for w in words for w_i in w])
    char2idx = {c: i + 2 for i, c in enumerate(chars)}
    char2idx["UNK"] = 1
    char2idx["PAD"] = 0
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char, len(chars), chars


def get_word_to_index_mappings(full_data):
    words = list(set(full_data.iloc[:, 0].index))
    word2idx = {w: i + 2 for i, w in enumerate(words)}
    word2idx["UNK"] = 1  # Unknown words
    word2idx["PAD"] = 0  # Padding
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word, len(words), words


def get_tag_to_index_mappings(full_data):
    tags = list(set(full_data.iloc[:, 0].values))
    tag2idx = {t: i + 1 for i, t in enumerate(tags)}
    tag2idx["PAD"] = 0
    idx2tag = {i: w for w, i in tag2idx.items()}
    return tag2idx, idx2tag, len(tags), tags


def read_df_from_tsv_file(file_path):
    return pd.read_csv(file_path, encoding="latin1", delimiter='\t').fillna(method="ffill")


def read_folds(task_name, which_fold, bpath="data"):
    full_path = os.path.join(bpath, "full-{task_name}.txt".format(task_name=task_name))
    train_path = os.path.join(bpath, "folds", task_name, str(which_fold), "train.tsv")
    test_path = os.path.join(bpath, "folds", task_name, str(which_fold), "test.tsv")
    return read_df_from_tsv_file(full_path), \
           read_df_from_tsv_file(train_path), \
           read_df_from_tsv_file(test_path)


def read_data_for_task(task_name, bpath="data"):
    full_path = os.path.join(bpath, "full-{task_name}.txt".format(task_name=task_name))
    train_path = os.path.join(bpath, "train-{task_name}.txt".format(task_name=task_name))
    test_path = os.path.join(bpath, "test-{task_name}.txt".format(task_name=task_name))
    return read_df_from_tsv_file(full_path), \
           read_df_from_tsv_file(train_path), \
           read_df_from_tsv_file(test_path)


def pad_string_matrix(string_matrix, max_len, pad_value='__PAD__'):
    padded_matrix = []
    for seq in string_matrix:
        new_seq = []
        for i in range(max_len):
            try:
                new_seq.append(seq[i])
            except:
                new_seq.append(pad_value)
        padded_matrix.append(new_seq)
    return padded_matrix
