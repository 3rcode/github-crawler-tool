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
    total = len(message)
    train_commit = message[:int(total * 0.8)]
    train_importance = label[:int(total * 0.8)]
    test_commit = message[int(total * 0.8):]
    test_importance = label[int(total * 0.8):]

    return (train_commit, train_importance), (test_commit, test_importance)


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
    vector = []
    for word in processed_commit:
        if word in word2idx:
            vector.append(word2idx[word])
        else:
            print("something wrong")
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
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    (train_data, train_labels), (test_data, test_labels) = load_data('/content/drive/MyDrive/data.csv')
    train_data = [preprocess(commit) for commit in train_data]
    test_data = [preprocess(commit) for commit in test_data]
    print(len(train_data), len(test_data))
    create_dictionary(train_data)
    create_dictionary(test_data)
    train_data = [convert2vector(commit) for commit in train_data]
    test_data = [convert2vector(commit) for commit in test_data]

    max_commit_length = 100
    train_data = np.array(
        [vector[:100] if len(vector) >= 100 else [0] * (100 - len(vector)) + vector for vector in train_data])
    test_data = np.array(
        [vector[:100] if len(vector) >= 100 else [0] * (100 - len(vector)) + vector for vector in test_data])
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    top_words = 4000
    embedding_vector_length = 32
    print(train_data[0])
    classify_model = build_model(top_words, embedding_vector_length, max_commit_length)
    print(classify_model.summary())
    classify_model.fit(train_data, train_labels, epochs=3, batch_size=64)
    scores = classify_model.evaluate(test_data, test_labels, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))











