import numpy as np
import pandas as pd
from tensorflow import dtypes
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF

data = pd.read_csv('../Dataset/Ansar1.txt', sep='\t', lineterminator='\r')
data = data.sort_values('ThreadID')

ner_data = pd.read_csv("../Dataset/ner_dataset.csv", encoding="latin1")
ner_data = ner_data.fillna(method="ffill")
print(ner_data.tail(10))


class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


def getThreads(posts):  # posts is a dataframe
    posts = posts.to_numpy()
    threads = []
    threadId = -1
    for i in range(posts.shape[0]):
        if posts[i][1] == threadId:
            threads[len(threads) - 1].append(posts[i])
        else:
            threadId = posts[i][1]
            threads.append([posts[i]])
    threads = np.asarray([np.array(thread) for thread in threads])  # convert 3d matrix to numpy array
    return threads


def preprocess_training_data():
    words = list(set(ner_data["Word"].values))
    words.append("ENDPAD")
    n_words = len(words)
    print(n_words)

    tags = list(set(ner_data["Tag"].values))
    n_tags = len(tags)
    print(n_tags)

    getter = SentenceGetter(ner_data)
    sent = getter.get_next()
    print(sent)
    sentences = getter.sentences
    max_len = 75
    word2idx = {w: i + 1 for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}
    print(word2idx["Obama"])
    print(tag2idx["B-geo"])

    # tokenize and prepare the sentences
    X = [[word2idx[w[0]] for w in s] for s in sentences]
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=0)

    # tags
    y = [[tag2idx[w[2]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])
    y = [to_categorical(i, num_classes=n_tags) for i in y]

    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)

    return X_tr, X_te, y_tr, y_te, max_len, n_words, n_tags


def train_model():
    m_input = Input(shape=(max_len,))
    model = Embedding(input_dim=n_words + 1, output_dim=20,
                      input_length=max_len, mask_zero=True)(m_input)  # 20-dim embedding
    model = Bidirectional(LSTM(units=50, return_sequences=True,
                               recurrent_dropout=0.1))(model)  # variational biLSTM
    model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
    crf = CRF(n_tags)  # CRF layer
    out = crf(model)  # output

    model = Model(m_input, out)

    model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

    print(model.summary())
    model.fit(X_tr, np.array(y_tr), batch_size=32, epochs=5,
              validation_split=0.1, verbose=1)

    return model


def run_bilstm(threads):
    print(threads[0][0][5])


threads = getThreads(data)
X_tr, X_te, y_tr, y_te, max_len, n_words, n_tags = preprocess_training_data()
t_model, t_history = train_model()
run_bilstm(threads)
