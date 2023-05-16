import spacy
# import nltk
import numpy as np
import pandas as pd
import os
# import re
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load('en_core_web_lg')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
word2idx = {}
max_commit_length = 30


def load_data(file_path):
    """Load the dataset from file.

    Returns
    -------
    tuple
        (train_data, train_labels), (test_data, test_labels).

    """
    df = pd.read_csv(file_path)
    commit = df['Commit Message'].astype(str) + df['Commit Description'].astype(str)
    label = df['Class']
    return np.asarray(commit), np.asarray(label)


def create_dictionary(commits):
    """Create a dictionary of words from commits.

    Parameters
    ----------
    commits : list
        List of commits.

    Returns
    -------
    dict
        Dictionary of words.

    """
    word_count = {}
    for commit in commits:
        lst_words = commit.split(' ')
        for word in lst_words:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
    sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    global word2idx
    for idx, (word, count) in enumerate(sorted_word_count):
        word2idx[word] = idx + 1
    word2idx['unk'] = len(word2idx) + 1


def convert2vector(commit):
    """Convert commit to a vector.

    Parameters
    ----------
    commit: str
        Commit.

    Returns
    -------
    list
        Vector of the commit.

    """
    global word2idx
    vector = []
    commit = commit.split(' ')
    for word in commit:
        if word in word2idx:
            vector.append(word2idx[word])
        else:
            vector.append(word2idx['unk'])
    if len(vector) >= max_commit_length:
        vector = vector[:max_commit_length]
    else:
        vector = [0] * (max_commit_length - len(vector)) + vector
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

    repos = []
    data_path = os.path.join(ROOT_DIR, 'data')
    for subdir, dirs, files in os.walk(data_path):
        repos.extend(dirs)
    X = np.asarray([])
    y = np.asarray([])
    for repo in repos:
        path = os.path.join(data_path, repo, 'labeled_commits.csv')
        commit, label = load_data(path)
        X = np.concatenate((X, commit), axis=0)
        y = np.concatenate((y, label), axis=0)
    # Check number of dataset
    # print(X.shape, y.shape)   
    def LSTM_model(X, y):
        create_dictionary(X)
        # Check create_dictionary function
        print(f"Length of dictionary: {len(word2idx)}")

        X = np.asarray([convert2vector(commit) for commit in X])
        # Check convert to vector function
        print(X.shape)
        print(X[0])

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

        top_words = len(word2idx) + 1
        embedding_vector_length = 300
        # classify_model = build_model(top_words, embedding_vector_length, max_commit_length)
        classify_model = load_model('models/lstm_model')
        # Check build_model function
        print(classify_model.summary())
        # classify_model.fit(X_train, y_train, epochs=3, batch_size=64) 
        # classify_model.save('models/lstm_model')
        scores = classify_model.evaluate(X_val, y_val, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))
        preds = classify_model.predict(X_val)
        preds = [1 if x > 0.5 else 0 for x in preds]
        score = f1_score(y_val, preds)
        print(f"F1 score: {score}")
    
    def naive_bayes(X, y):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(X)
        print(f"Num Samples: {X.shape[0]}\nNum Features: {X.shape[1]}")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
        model = MultinomialNB()
        model.fit(X_train, y_train)
        test_size = X_val.shape[0]
        y_pred = model.predict(X_val)
        true_pred = sum([y_pred[i] == y_val[i] for i in range(test_size)])
        print("Accuracy: %.2f%%" % (true_pred / test_size * 100))
        score = f1_score(y_val, y_pred)
        print(f"F1 score: {score}")
    
    naive_bayes(X, y)
    


