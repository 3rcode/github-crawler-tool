import spacy
import nltk
import numpy as np
import pandas as pd
import re
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from tensorflow import keras

nlp = spacy.load('en_core_web_lg')
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
word2idx = {}
idx2word = {}


def load_data(file_path):
    """Load the dataset from file.

    Returns
    -------
    tuple
        (train_data, train_labels), (test_data, test_labels).

    """
    df = pd.read_csv(file_path)
    message = np.asarray(df['message'])
    label = np.asarray(df['is_important'])
    label = np.asarray(list(map(lambda x: 1 if x == 1 else 0, label)))

    return message, label


def separate_data(data, ratio):
    """Separate the data into training and testing sets.

    Parameters
    ----------
    data : tuple
        (message, labels).
    ratio : float
        Ratio of training data to testing data.

    Returns
    -------
    tuple
        (train_data, train_labels), (test_data, test_labels).

    """
    data_size = len(data[0])
    train_size = int(data_size * ratio)
    train_dat = (data[0][:train_size], data[1][:train_size])
    test_dat = (data[0][train_size:], data[1][train_size:])
    return train_dat, test_dat


def preprocess(commit, remove_stopwords=True):
    """Preprocess a commit message.

    Parameters
    ----------
    commit : str
        Commit message.
    remove_stopwords : bool, optional
        Remove stopwords from the commit message. The default is True.

    Returns
    -------
    str
        Preprocessed commit message.

    """
    global nlp, stopwords
    commit = commit.lower()
    # Format words and remove unwanted characters
    commit = re.sub(r'https?://.*[\r\n]*', '', commit, flags=re.MULTILINE)
    commit = re.sub(r'<a href', ' ', commit)
    commit = re.sub(r'&amp;', '', commit)
    commit = re.sub(r'[_"\-;%()|+&=*.,!?:#$@\[\]/]', ' ', commit)
    commit = re.sub(r'<br />', ' ', commit)
    commit = re.sub(r'\'', ' ', commit)
    # Optionally, remove stop words
    if remove_stopwords:
        global stopwords
        commit = commit.split()
        commit = [w for w in commit if w not in stopwords]
        commit = " ".join(commit)
    nlp_commit = nlp(commit)
    processed_commit = [token.text for token in nlp_commit]
    return processed_commit


def create_dictionary(all_processed_commits):
    """Create a dictionary of words from all processed commits.

    Parameters
    ----------
    all_processed_commits : list
        List of processed commits.

    Returns
    -------
    dict
        Dictionary of words.

    """
    word_count = {}
    for processed_commit in all_processed_commits:
        for word in processed_commit:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
    sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    global word2idx
    global idx2word
    for idx, (word, count) in enumerate(sorted_word_count):
        word2idx[word] = idx
        idx2word[idx] = word


def convert2vector(processed_commit):
    """Convert a processed commit to a vector.

    Parameters
    ----------
    processed_commit : list
        Processed commit.

    Returns
    -------
    list
        Vector of the processed commit.

    """
    global word2idx
    global idx2word
    vector = []
    for word in processed_commit:
        if word in word2idx:
            vector.append(word2idx[word])
        else:
            word = "unk"
            vector.append(word2idx[word])
    return vector


def build_model(topwords, embedding_vector_len, input_length):
    """Build the model.

    Returns
    -------
    keras.Sequential
        Model.

    """
    model = Sequential()
    model.add(Embedding(topwords, embedding_vector_len, input_length=input_length))
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    (train_data, train_labels), (test_data, test_labels) = separate_data(load_data('data.csv'), 0.8)
    train_data = [preprocess(commit) for commit in train_data]
    test_data = [preprocess(commit) for commit in test_data]
    create_dictionary(train_data)
    create_dictionary(test_data)
    word2idx["unk"] = len(word2idx)
    idx2word[len(idx2word)] = "unk"
    train_data = [convert2vector(commit) for commit in train_data]
    test_data = [convert2vector(commit) for commit in test_data]

    max_commit_length = 100
    train_data = np.array(
        [vector[:100] if len(vector) >= 100 else [0] * (100 - len(vector)) + vector for vector in train_data])
    test_data = np.array(
        [vector[:100] if len(vector) >= 100 else [0] * (100 - len(vector)) + vector for vector in test_data])
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    print(f"Length of dictionary: {len(word2idx)}")
    top_words = len(word2idx) + 1
    embedding_vector_length = 300
    classify_model = build_model(top_words, embedding_vector_length, max_commit_length)
    print(classify_model.summary())
    classify_model.fit(train_data, train_labels, epochs=3, batch_size=64)
    scores = classify_model.evaluate(test_data, test_labels, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    mes, res = load_data('test.csv')
    mes = [preprocess(commit) for commit in mes]
    mes = [convert2vector(commit) for commit in mes]
    mes = np.array([vector[:100] if len(vector) >= 100 else [0] * (100 - len(vector)) + vector for vector in mes])
    predict = classify_model.predict(mes)
    count_success_case = 0
    for i in range(len(mes)):
        print("Message: {}\tPredict: {}\t Actual: {}".format(mes[i], predict[i], res[i]))
        if (predict[i] < 0.5 and res[i] == 0) or (predict[i] >= 0.5 and res[i] == 1):
            count_success_case += 1
    print("Accuracy: {}".format(count_success_case / len(mes)))


